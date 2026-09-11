"""Pick up a box, then walk backward away from the table with the payload."""

import argparse
import math

import mujoco
import numpy as np

from ..config import CARRY, PALM
from ..geometry import blend
from ..metrics import CarryStats as CarryStats
from ..paths import ROOT
from ..runner import payload_camera_target, positive, run_experiment
from ..scenes import hold_model
from .pickup import Pickup


class Carry(Pickup):
    follow_camera = True
    carry_config = CARRY

    def __init__(self, speed=CARRY.speed, assets="amazon", squeeze=PALM.squeeze):
        if not math.isfinite(speed) or not 0 < speed <= self.carry_config.max_speed:
            raise ValueError("Backward speed must be finite and in (0, 0.3] m/s")
        if assets not in ("amazon", "simple"):
            raise ValueError("Assets must be amazon or simple")
        self.speed = speed
        self.assets = assets
        model = (
            hold_model(scene=ROOT / "assets" / "amazon_pickup.xml") if assets == "amazon" else None
        )
        super().__init__(squeeze=squeeze, model=model)

    def reset(self):
        super().reset()
        # Walking lowers the pelvis; leave extra clearance above the tabletop.
        self.lift_delta[2] = self.carry_config.lift_height
        self.carry_offsets = None
        self.walk_origin = None
        self.box_origin = None
        self.grasp_origin = None
        self.carry_rotation = None
        self.carry_grasp_rotations = None
        self.foot_ids = [self.model.body(f"{side}-toe-roll").id for side in ("left", "right")]
        self.floor_id = self.model.geom("floor").id

    def begin_carry(self):
        rotation = self.yaw_rotation()
        self.carry_offsets = (self.targets - self.data.qpos[:3]) @ rotation
        self.walk_origin = self.data.qpos[:3].copy()
        self.box_origin = self.data.qpos[self.box_adr : self.box_adr + 3].copy()
        self.grasp_origin = self.measure()["box_in_grasp"].copy()
        self.carry_rotation = rotation.copy()
        self.carry_grasp_rotations = self.grasp_rotations.copy()
        self.transition("carry")
        self.controller.set_mode(2)

    def update_task(self):
        if self.phase != "carry":
            super().update_task()
            if (
                self.phase == "hold"
                and self.data.time - self.phase_start >= self.carry_config.hold_seconds
            ):
                values = self.measure()
                ready = (
                    np.all(
                        values["normal_force"]
                        > self.pickup_config.friction_margin
                        * self.payload_weight
                        / (2 * self.palm_config.friction)
                    )
                    and values["other_contacts"] == 0
                    and values["platform_clearance_m"] > self.carry_config.minimum_clearance
                    and values["tilt_degrees"] < self.carry_config.ready_tilt_degrees
                )
                if ready:
                    self.begin_carry()
                elif self.data.time - self.phase_start > self.carry_config.ready_timeout_seconds:
                    raise RuntimeError(
                        "Payload is not securely held clear of the table; refusing to walk"
                    )
        if self.phase == "carry":
            elapsed = self.data.time - self.phase_start
            velocity = -self.speed * float(
                blend(0, 1, elapsed, self.carry_config.speed_ramp_seconds)[0]
            )
            self.controller.set_velocity(velocity, 0, 0)
            rotation = self.yaw_rotation()
            offsets_world = self.carry_offsets @ rotation.T
            self.targets = self.data.qpos[:3] + offsets_world
            # Follow translation/yaw, keeping the payload level through torso roll/pitch.
            body_rotation = self.data.xmat[self.base_id].reshape(3, 3)
            angular = body_rotation @ self.data.qvel[3:6]
            forward = body_rotation[:, 0]
            forward_rate = np.cross(angular, forward)
            yaw_rate = (forward[0] * forward_rate[1] - forward[1] * forward_rate[0]) / max(
                np.dot(forward[:2], forward[:2]), 1e-6
            )
            self.target_velocities = self.data.qvel[:3] + np.cross([0, 0, yaw_rate], offsets_world)
            self.grasp_rotations = rotation @ self.carry_rotation.T @ self.carry_grasp_rotations
            values = self.measure()
            if values["other_contacts"]:
                raise RuntimeError("Payload touched a surface outside the palms while carrying")
            if np.all(values["normal_force"] > self.palm_config.minimum_contact_force):
                self.contact_lost_since = None
            elif self.contact_lost_since is None:
                self.contact_lost_since = self.data.time
            elif self.data.time - self.contact_lost_since > self.palm_config.contact_loss_seconds:
                raise RuntimeError("Grasp lost while carrying")

    def release(self):
        self.controller.set_velocity(0, 0, 0)
        super().release()

    def palm_command(self, side):
        if self.phase != "carry":
            return super().palm_command(side)
        inward = self.yaw_rotation() @ [0, -self.squeeze if side == 0 else self.squeeze, 0]
        force = inward + [0, 0, self.payload_weight / 2]
        return self.targets[side], self.target_velocities[side], force

    def camera_target(self):
        return payload_camera_target(self)

    def measure(self):
        values = super().measure()
        m, d = self.model, self.data
        box = d.qpos[self.box_adr : self.box_adr + 3]
        palms_midpoint = d.site_xpos[self.sites].mean(axis=0)
        foot_loads = np.zeros(2)
        wrench = np.zeros(6)
        for i, contact in enumerate(d.contact):
            if self.floor_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
            body = m.geom_bodyid[other]
            if body in self.foot_ids:
                mujoco.mj_contactForce(m, d, i, wrench)
                foot_loads[self.foot_ids.index(body)] += wrench[0]
        table_rear = m.geom_pos[self.platform, 0] - m.geom_size[self.platform, 0]
        box_front = box[0] + np.abs(d.xmat[self.box_id].reshape(3, 3)[0]) @ self.box_halfsize
        values.update(
            box_position=box.copy(),
            base_position=d.qpos[:3].copy(),
            box_in_grasp=self.yaw_rotation().T @ (box - palms_midpoint),
            foot_loads=foot_loads,
            table_separation_m=float(table_rear - box_front),
        )
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration", type=positive, default=30, help="simulation seconds, at least 20"
    )
    parser.add_argument(
        "--speed",
        type=positive,
        default=CARRY.speed,
        help="backward velocity target, at most .3 m/s",
    )
    parser.add_argument("--assets", choices=("amazon", "simple"), default="amazon")
    parser.add_argument(
        "--squeeze", type=positive, default=PALM.squeeze, help="inward force per palm in newtons"
    )
    parser.add_argument("--contacts", action="store_true")
    args = parser.parse_args()
    if args.duration < 20:
        parser.error("--duration must be at least 20 seconds to validate pickup and carrying")
    if args.speed > CARRY.max_speed:
        parser.error("--speed must be at most .3 m/s for this experiment")
    run_experiment(Carry(args.speed, args.assets, args.squeeze), CarryStats(), args)


if __name__ == "__main__":
    main()
