"""Move the top Amazon box from the original pile to the original table."""

import argparse
import math

import numpy as np

from ..config import TRANSFER
from ..geometry import blend
from ..manipulation import PalmImpedance
from ..metrics import TransferStats as TransferStats
from ..runner import payload_camera_target, positive, run_experiment
from ..scenes import transfer_model
from .pickup import Pickup


class Transfer(Pickup):
    follow_camera = True
    center_robot = False
    transfer_config = TRANSFER

    def __init__(self, *, overview=False, speed_scale=TRANSFER.speed_scale):
        if (
            not math.isfinite(speed_scale)
            or not 1 <= speed_scale <= self.transfer_config.max_speed_scale
        ):
            raise ValueError("Speed scale must be finite and between 1 and 2")
        self.speed_scale = speed_scale
        self.overview = overview
        self.follow_camera = not overview
        self.camera_distance = 7 if overview else 2.5
        super().__init__(model=transfer_model())

    def camera_target(self):
        return np.array([1.6, -0.8, 0.75]) if self.overview else payload_camera_target(self)

    def reset(self):
        # Initialize the robot and task directly, preserving all scene object poses.
        PalmImpedance.reset(self)
        self.initialize_pickup(place_on_platform=False)
        self.targets = self.data.site_xpos[self.sites].copy()
        self.start_palms = self.targets.copy()
        self.travel_offsets = (self.targets - self.data.qpos[:3]) @ self.yaw_rotation()
        self.filtered_velocity = np.zeros(2)
        self.integral = np.zeros(2)
        self.destination = np.array(self.transfer_config.pickup_waypoint)
        self.lift_delta = np.array([0, 0, self.transfer_config.lift_height])
        self.place_center = np.array(self.transfer_config.place_center)
        self.minimum_base = 10.0
        self.foot_ids = [self.model.body(f"{side}-toe-roll").id for side in ("left", "right")]
        self.floor_id = self.model.geom("floor").id
        self.grasp_moment = None
        self.grasp_axis = np.array([0.0, 1.0, 0.0])

    def palm_command(self, side):
        target, velocity, force = super().palm_command(side)
        desired_rotation = self.yaw_rotation()
        if self.phase in ("settle-at-table", "lower"):
            # Match orientation/squeeze to the hand line as it straightens for placement.
            axis = self.targets[0] - self.targets[1]
            axis[2] = 0
            axis /= np.linalg.norm(axis)
            self.grasp_axis = axis
            desired_rotation = np.column_stack(([axis[1], -axis[0], 0], axis, [0, 0, 1]))
        if self.phase in ("clear-pile", "walk-to-table", "settle-at-table", "lower"):
            force = desired_rotation @ force
        if self.phase in (
            "lift",
            "hold",
            "clear-pile",
            "walk-to-table",
            "settle-at-table",
            "lower",
        ):
            rotation = self.data.xmat[self.box_id].reshape(3, 3)
            error = 0.5 * sum(np.cross(rotation[:, k], desired_rotation[:, k]) for k in range(3))
            box_dof = self.model.joint("held-box").dofadr[0]
            angular = rotation @ self.data.qvel[box_dof + 3 : box_dof + 6]
            self.grasp_moment = float(
                self.transfer_config.pitch_gain * np.dot(error, self.grasp_axis)
                - self.transfer_config.pitch_damping * np.dot(angular, self.grasp_axis)
            )
            moment = (
                self.transfer_config.level_gain * error
                - self.transfer_config.level_damping * angular
            )
            correction = np.clip(
                np.array([-moment[2], 0, moment[0]]) / (2 * self.transfer_config.grasp_half_span),
                -self.transfer_config.level_force_limit,
                self.transfer_config.level_force_limit,
            )
            force = force + (1 if side == 0 else -1) * correction
        return target, velocity, force

    def orientation_moment(self, error, angular_velocity):
        if self.grasp_moment is None:
            return super().orientation_moment(error, angular_velocity)
        return self.grasp_axis * self.grasp_moment

    def motor_torques(self):
        torque = super().motor_torques()
        if self.phase in ("walk-to-pile", "clear-pile", "walk-to-table"):
            rotation = self.yaw_rotation()
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            angular = self.data.xmat[self.base_id].reshape(3, 3) @ self.data.qvel[3:6]
            steering = self.speed_scale if self.phase != "walk-to-pile" else 1
            moment = np.clip(
                -self.transfer_config.heading_gain * steering * yaw
                - self.transfer_config.heading_damping * steering * angular[2],
                -self.transfer_config.heading_moment_limit * steering,
                self.transfer_config.heading_moment_limit * steering,
            )
            # Steer the feet toward the route heading; upstream fixes hip yaw at zero.
            grounded = set()
            for contact in self.data.contact:
                if self.floor_id in (contact.geom1, contact.geom2) and contact.dist < 0:
                    other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
                    grounded.add(self.model.geom_bodyid[other])
            for side, idx in enumerate((1, 7)):
                if self.foot_ids[side] not in grounded:
                    torque[idx] += 0.5 * moment * self.data.xaxis[self.joint_ids[idx], 2]
        return torque

    def start_walk(self, phase, destination):
        self.destination = np.array(destination)
        self.integral[:] = 0
        self.travel_offsets = (self.targets - self.data.qpos[:3]) @ self.yaw_rotation()
        self.travel_rotation = self.yaw_rotation().copy()
        self.travel_grasp = None if self.grasp_rotations is None else self.grasp_rotations.copy()
        self.controller.set_mode(2)
        self.transition(phase)

    def navigate(self):
        dt = self.model.opt.timestep
        self.filtered_velocity += (
            dt
            / self.transfer_config.velocity_filter_seconds
            * (self.data.qvel[:2] - self.filtered_velocity)
        )
        error = self.destination - self.data.qpos[:2]
        self.integral = np.clip(
            self.integral + self.transfer_config.integral_gain * dt * error,
            -self.transfer_config.integral_limit,
            self.transfer_config.integral_limit,
        )
        # Preserve the calibrated approach/stop at the pile; accelerate transport.
        scale = self.speed_scale if self.phase in ("clear-pile", "walk-to-table") else 1
        command = np.clip(
            self.transfer_config.position_gain * error
            - self.transfer_config.velocity_damping * self.filtered_velocity
            + self.integral,
            -self.transfer_config.forward_speed * scale,
            self.transfer_config.forward_speed * scale,
        )
        command[1] = np.clip(
            command[1],
            -self.transfer_config.lateral_speed * scale,
            self.transfer_config.lateral_speed * scale,
        )
        rotation = self.yaw_rotation()
        self.grasp_axis = rotation[:, 1].copy()
        local = rotation[:2, :2].T @ command
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.controller.set_velocity(
            float(local[0]), float(local[1]), float(-self.transfer_config.turn_gain * yaw)
        )
        if self.travel_grasp is not None:
            self.grasp_rotations = rotation @ self.travel_rotation.T @ self.travel_grasp
        offsets = self.travel_offsets @ rotation.T
        self.targets = self.data.qpos[:3] + offsets
        self.target_velocities = np.tile(self.data.qvel[:3], (2, 1))
        return (
            np.linalg.norm(error) < self.transfer_config.arrival_radius
            and np.linalg.norm(self.filtered_velocity) < self.transfer_config.arrival_speed
        )

    def stop_walk(self, phase):
        self.controller.set_velocity(0, 0, 0)
        self.controller.set_mode(0)
        self.target_velocities[:] = 0
        self.transition(phase)

    def update_task(self):
        t = self.data.time
        elapsed = t - self.phase_start
        self.minimum_base = min(self.minimum_base, float(self.data.qpos[2]))
        if self.minimum_base < self.transfer_config.minimum_base_height:
            raise RuntimeError(f"Robot fell during {self.phase}")
        if self.phase == "settle":
            if elapsed >= self.transfer_config.settle_seconds:
                self.start_walk("walk-to-pile", self.transfer_config.pickup_waypoint)
        elif self.phase == "walk-to-pile":
            if self.navigate():
                self.stop_walk("settle-at-pile")
        elif self.phase == "settle-at-pile":
            if elapsed >= self.transfer_config.settle_seconds:
                self.pick_center = self.data.qpos[self.box_adr : self.box_adr + 3].copy()
                self.start_palms = self.data.site_xpos[self.sites].copy()
                self.closed_palms = np.array(
                    [
                        self.pick_center + [0, s * self.transfer_config.grasp_half_span, 0]
                        for s in (1, -1)
                    ]
                )
                self.open_palms = self.closed_palms + [
                    [0, self.pickup_config.open_clearance, 0],
                    [0, -self.pickup_config.open_clearance, 0],
                ]
                self.transition("approach")
        elif self.phase in ("approach", "close", "grasp", "lift", "hold"):
            super().update_task()
            if self.phase == "hold" and t - self.phase_start >= self.transfer_config.hold_seconds:
                self.start_walk("clear-pile", [self.data.qpos[0], self.transfer_config.aisle_y])
        elif self.phase in ("clear-pile", "walk-to-table"):
            arrived = self.navigate()
            if self.phase == "clear-pile" and arrived:
                self.destination = np.array(self.transfer_config.table_waypoint)
                self.integral[:] = 0
                self.transition("walk-to-table")
            elif self.phase == "walk-to-table" and arrived:
                self.stop_walk("settle-at-table")
        elif self.phase == "settle-at-table":
            if elapsed >= self.transfer_config.settle_seconds:
                self.lower_start = self.targets.copy()
                self.lower_end = np.array(
                    [
                        self.place_center
                        + [
                            0,
                            s * self.transfer_config.grasp_half_span,
                            -self.transfer_config.table_penetration,
                        ]
                        for s in (1, -1)
                    ]
                )
                self.transition("lower")
        elif self.phase == "lower":
            self.targets, self.target_velocities = blend(
                self.lower_start, self.lower_end, elapsed, self.transfer_config.lower_seconds
            )
            self.support_scale = 1 - float(
                blend(0, 1, elapsed, self.transfer_config.lower_seconds)[0]
            )
            if (
                elapsed >= self.transfer_config.lower_seconds
                and self.measure()["platform_force_n"]
                > self.transfer_config.table_support_fraction * self.payload_weight
            ):
                self.transition("open")
                self.open_start = self.targets.copy()
            elif elapsed > self.transfer_config.lower_timeout_seconds:
                raise RuntimeError("Table did not support the payload")
        elif self.phase == "open":
            a = float(blend(0, 1, elapsed, self.transfer_config.open_seconds)[0])
            self.squeeze_scale = 1 - a
            self.targets, self.target_velocities = blend(
                self.open_start,
                self.open_start
                + [
                    [0, self.palm_config.release_clearance, 0],
                    [0, -self.palm_config.release_clearance, 0],
                ],
                elapsed,
                self.transfer_config.open_seconds,
            )
            self.grasp_rotations = None
            if elapsed >= self.transfer_config.open_seconds:
                self.retract_start = self.targets.copy()
                self.transition("retract")
        elif self.phase == "retract":
            self.targets, self.target_velocities = blend(
                self.retract_start,
                self.retract_start + self.transfer_config.retract_delta,
                elapsed,
                self.transfer_config.retract_seconds,
            )
            if elapsed >= self.transfer_config.retract_seconds:
                self.transition("done")
        if self.phase in ("clear-pile", "walk-to-table", "settle-at-table"):
            values = self.measure()
            if values["other_contacts"]:
                raise RuntimeError(f"Payload collided during {self.phase}")
            if np.all(values["normal_force"] > self.palm_config.minimum_contact_force):
                self.contact_lost_since = None
            elif self.contact_lost_since is None:
                self.contact_lost_since = t
            elif t - self.contact_lost_since > self.palm_config.contact_loss_seconds:
                raise RuntimeError(f"Grasp lost during {self.phase}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=positive, default=100)
    parser.add_argument("--contacts", action="store_true")
    parser.add_argument(
        "--overview", action="store_true", help="show the pile, route, and table together"
    )
    parser.add_argument(
        "--speed-scale",
        type=positive,
        default=TRANSFER.speed_scale,
        help="carrying speed multiplier, 1 to 2; 1 restores the original pace",
    )
    args = parser.parse_args()
    if not 1 <= args.speed_scale <= TRANSFER.max_speed_scale:
        parser.error("--speed-scale must be between 1 and 2")
    run_experiment(
        Transfer(overview=args.overview, speed_scale=args.speed_scale), TransferStats(), args
    )


if __name__ == "__main__":
    main()
