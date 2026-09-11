"""Approach, grasp, and lift a free box from a fixed platform."""

import argparse

import mujoco
import numpy as np

from ..config import PALM, PICKUP
from ..geometry import blend
from ..manipulation import PalmImpedance
from ..metrics import PickupStats as PickupStats
from ..robot import INITIAL_QPOS
from ..runner import positive, run_experiment
from ..scenes import hold_model


class Pickup(PalmImpedance):
    pickup_config = PICKUP

    def __init__(self, squeeze=PALM.squeeze, *, model=None):
        super().__init__(
            squeeze=squeeze, model=hold_model(platform=True) if model is None else model
        )

    def reset(self):
        super().reset()
        self.initialize_pickup(place_on_platform=True)

    def initialize_pickup(self, *, place_on_platform):
        """Initialize task targets; only platform pickup repositions the box."""
        m, d = self.model, self.data
        self.platform = m.geom("pickup-platform").id
        self.platform_top = m.geom_pos[self.platform, 2] + m.geom_size[self.platform, 2]
        self.box_halfsize = m.geom_size[m.geom("held-box").id].copy()
        self.pick_center = np.array(
            [self.pickup_config.center_x, 0, self.platform_top + self.box_halfsize[2]]
        )
        # Reset starts with the box supported by the platform and arms at rest.
        # Subsequent motion uses motor torques only.
        if place_on_platform:
            d.qpos[self.box_adr : self.box_adr + 3] = self.pick_center
            d.qpos[self.box_adr + 3 : self.box_adr + 7] = [1, 0, 0, 0]
        else:
            self.pick_center = d.qpos[self.box_adr : self.box_adr + 3].copy()
        d.qpos[self.qadr[12:20]] = INITIAL_QPOS[self.qadr[12:20]]
        mujoco.mj_forward(m, d)
        self.posture = d.qpos[self.qadr[12:20]].copy()
        self.start_palms = d.site_xpos[self.sites].copy()
        half_span = self.offsets[0][1]
        self.open_palms = np.array(
            [
                self.pick_center + [0, sign * (half_span + self.pickup_config.open_clearance), 0]
                for sign in (1, -1)
            ]
        )
        self.closed_palms = np.array(
            [self.pick_center + [0, sign * half_span, 0] for sign in (1, -1)]
        )
        self.lift_delta = np.array([0.0, 0, self.pickup_config.lift_height])
        self.orientation_gain = self.pickup_config.orientation_gain
        self.grasp_rotations = None
        self.rot_jac = np.zeros((3, m.nv))
        self.phase = "settle"
        self.phase_start = 0.0
        self.contact_since = None
        self.contact_lost_since = None
        self.events = [("settle", 0.0)]
        self.targets = self.start_palms.copy()
        self.target_velocities = np.zeros((2, 3))
        self.squeeze_scale = 0.0
        self.support_scale = 0.0

    def transition(self, phase):
        self.phase = phase
        self.phase_start = self.data.time
        self.events.append((phase, self.data.time))
        if phase == "lift":
            self.grasp_rotations = self.data.site_xmat[self.sites].copy().reshape(2, 3, 3)

    def update_task(self):
        t = self.data.time
        elapsed = t - self.phase_start
        if self.phase == "settle" and elapsed >= self.pickup_config.settle_seconds:
            self.transition("approach")
        elif self.phase == "approach" and elapsed >= self.pickup_config.approach_seconds:
            self.transition("close")
        elif self.phase == "close" and elapsed >= self.pickup_config.close_seconds:
            self.transition("grasp")
        elif self.phase == "grasp":
            # Require a friction margin above the static payload weight.
            required_normal = (
                self.pickup_config.friction_margin
                * self.payload_weight
                / (2 * self.palm_config.friction)
            )
            contact = np.all(self.measure()["normal_force"] > required_normal)
            if not contact:
                self.contact_since = None
            elif self.contact_since is None:
                self.contact_since = t
            elif t - self.contact_since >= self.pickup_config.contact_seconds:
                self.transition("lift")
            if self.phase == "grasp" and elapsed > self.pickup_config.grasp_timeout_seconds:
                raise RuntimeError("Grasp failed: both palms must maintain contact before lifting")
        elif self.phase == "lift" and elapsed >= self.pickup_config.lift_seconds:
            self.transition("hold")
        if self.phase in ("lift", "hold"):
            if np.all(self.measure()["normal_force"] > self.palm_config.minimum_contact_force):
                self.contact_lost_since = None
            elif self.contact_lost_since is None:
                self.contact_lost_since = t
            elif t - self.contact_lost_since > self.palm_config.contact_loss_seconds:
                raise RuntimeError("Grasp lost during lift/hold")
        elapsed = t - self.phase_start
        self.target_velocities[:] = 0
        if self.phase == "settle":
            self.targets = self.start_palms.copy()
        elif self.phase == "approach":
            self.targets, self.target_velocities = blend(
                self.start_palms, self.open_palms, elapsed, self.pickup_config.approach_seconds
            )
        elif self.phase == "close":
            self.targets, self.target_velocities = blend(
                self.open_palms, self.closed_palms, elapsed, self.pickup_config.close_seconds
            )
            self.squeeze_scale = float(blend(0, 1, elapsed, self.pickup_config.close_seconds)[0])
        elif self.phase == "grasp":
            self.targets = self.closed_palms.copy()
            self.squeeze_scale = 1.0
        elif self.phase == "lift":
            self.targets, self.target_velocities = blend(
                self.closed_palms,
                self.closed_palms + self.lift_delta,
                elapsed,
                self.pickup_config.lift_seconds,
            )
            self.support_scale = float(
                blend(0, 1, elapsed, self.pickup_config.support_ramp_seconds)[0]
            )
        elif self.phase == "hold":
            self.targets = self.closed_palms + self.lift_delta
            self.support_scale = 1.0

    def release(self):
        super().release()
        self.transition("released")

    def palm_command(self, side):
        if self.released:
            target = self.targets[side] + [
                0,
                self.palm_config.release_clearance
                if side == 0
                else -self.palm_config.release_clearance,
                0,
            ]
            return target, np.zeros(3), np.zeros(3)
        force = np.array(
            [
                0,
                (-1 if side == 0 else 1) * self.squeeze * self.squeeze_scale,
                self.payload_weight / 2 * self.support_scale,
            ]
        )
        return self.targets[side], self.target_velocities[side], force

    def motor_torques(self):
        self.update_task()
        torque = super().motor_torques()
        if self.grasp_rotations is not None and not self.released:
            for side, site in enumerate(self.sites):
                idx = slice(12 + 4 * side, 16 + 4 * side)
                dofs = self.vadr[idx]
                mujoco.mj_jacSite(self.model, self.data, self.jac, self.rot_jac, site)
                jac = self.jac[:, dofs]
                null = np.eye(4) - np.linalg.pinv(jac) @ jac
                rotation = self.data.site_xmat[site].reshape(3, 3)
                error = 0.5 * sum(
                    np.cross(rotation[:, k], self.grasp_rotations[side, :, k]) for k in range(3)
                )
                angular_velocity = self.rot_jac @ self.data.qvel
                # Preserve twist about the grasp normal using the fourth arm DOF.
                # Projection keeps position control the primary task.
                moment = self.orientation_moment(error, angular_velocity)
                torque[idx] += null @ self.rot_jac[:, dofs].T @ moment
        return torque

    def orientation_moment(self, error, angular_velocity):
        """Desired twist moment about the palm-to-palm grasp axis."""
        return np.array(
            [
                0,
                self.orientation_gain * error[1]
                - self.pickup_config.orientation_damping * angular_velocity[1],
                0,
            ]
        )

    def measure(self):
        values = super().measure()
        m, d = self.model, self.data
        rotation = d.xmat[self.box_id].reshape(3, 3)
        bottom = values["box_height"] - np.abs(rotation[2]) @ self.box_halfsize
        platform_force = 0.0
        wrench = np.zeros(6)
        for i, contact in enumerate(d.contact):
            if self.platform not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.platform else contact.geom1
            if m.geom_bodyid[other] != self.box_id:
                continue
            mujoco.mj_contactForce(m, d, i, wrench)
            sign = 1 if contact.geom1 == self.platform else -1
            platform_force += sign * (contact.frame.reshape(3, 3).T @ wrench[:3])[2]
        values.update(
            platform_force_n=platform_force,
            platform_clearance_m=float(bottom - self.platform_top),
            phase=self.phase,
        )
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration", type=positive, default=20, help="simulation seconds, at least 12"
    )
    parser.add_argument(
        "--squeeze", type=positive, default=PALM.squeeze, help="inward force per palm in newtons"
    )
    parser.add_argument("--contacts", action="store_true")
    args = parser.parse_args()
    if args.duration < 12:
        parser.error("--duration must be at least 12 seconds for pickup and hold validation")
    run_experiment(Pickup(squeeze=args.squeeze), PickupStats(), args)


if __name__ == "__main__":
    main()
