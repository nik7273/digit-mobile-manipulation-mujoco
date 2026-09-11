"""Shared palm impedance, robot reset, and contact measurements."""

import math

import mujoco
import numpy as np

from .config import PALM
from .geometry import yaw_rotation
from .scenes import PALM_RADIUS
from .simulation import Simulation


class PalmImpedance(Simulation):
    center_robot = True
    palm_config = PALM

    def __init__(self, squeeze=PALM.squeeze, *, model):
        if not math.isfinite(squeeze) or squeeze < 0:
            raise ValueError("Squeeze force must be finite and nonnegative")
        self.squeeze = squeeze
        self.position_gain = self.palm_config.position_gain
        self.velocity_gain = self.palm_config.velocity_gain
        super().__init__(model=model, mode="standing-analytic")

    def reset(self):
        super().reset()
        m, d = self.model, self.data
        self.released = False
        if self.center_robot:
            d.qpos[0:2] = 0
        mujoco.mj_forward(m, d)
        self.sites = [m.site(f"{side}-palm").id for side in ("left", "right")]
        self.palms = [m.geom(f"{side}-palm").id for side in ("left", "right")]
        self.box_id = m.body("held-box").id
        self.payload_weight = m.body_mass[self.box_id] * abs(m.opt.gravity[2])
        self.box_adr = m.joint("held-box").qposadr[0]
        self.center_offset = np.array(self.palm_config.hold_offset)
        half_width = m.geom_size[m.geom("held-box").id, 1]
        self.offsets = [
            self.center_offset + [0, sign * (half_width + PALM_RADIUS), 0] for sign in (1, -1)
        ]
        self.jac = np.zeros((3, m.nv))
        self.base_jac = np.zeros((3, m.nv))
        self.posture = d.qpos[self.qadr[12:20]].copy()

    def yaw_rotation(self):
        return yaw_rotation(self.data.qpos[3:7])

    def camera_target(self):
        return np.array([0.15, 0, 0.9])

    def release(self):
        """Open the palms without changing the box state or applying box forces."""
        self.released = True

    def palm_command(self, side):
        """Desired world position/velocity and feedforward force for one palm."""
        m, d = self.model, self.data
        rotation = d.xmat[self.base_id].reshape(3, 3)
        offset = self.offsets[side].copy()
        if self.released:
            offset[1] += (
                self.palm_config.release_clearance
                if side == 0
                else -self.palm_config.release_clearance
            )
        target = d.xpos[self.base_id] + rotation @ offset
        mujoco.mj_jac(m, d, self.base_jac, None, target, self.base_id)
        velocity = self.base_jac @ d.qvel
        force = np.zeros(3)
        if not self.released:
            force += rotation @ np.array([0, -self.squeeze if side == 0 else self.squeeze, 0])
            force += np.array([0, 0, self.payload_weight / 2])
        return target, velocity, force

    def motor_torques(self):
        torque = super().motor_torques()
        m, d = self.model, self.data
        for side, site in enumerate(self.sites):
            idx = slice(12 + 4 * side, 16 + 4 * side)
            dofs = self.vadr[idx]
            target, velocity, feedforward = self.palm_command(side)
            mujoco.mj_jacSite(m, d, self.jac, None, site)
            # Damping uses motion relative to the desired trajectory.
            velocity_error = velocity - self.jac @ d.qvel
            force = (
                self.position_gain * (target - d.site_xpos[site])
                + self.velocity_gain * velocity_error
                + feedforward
            )
            jac = self.jac[:, dofs]
            # The fourth arm DOF softly prefers the initialized posture.
            null = np.eye(4) - np.linalg.pinv(jac) @ jac
            posture = (
                self.palm_config.posture_gain
                * (self.posture[side * 4 : side * 4 + 4] - d.qpos[self.qadr[idx]])
                - self.palm_config.posture_damping * d.qvel[dofs]
            )
            torque[idx] = jac.T @ force + d.qfrc_bias[dofs] + null @ posture
        return torque

    def measure(self):
        """Contact loads on the box; mj_contactForce is in the contact frame."""
        m, d = self.model, self.data
        normal = np.zeros(2)
        upward = np.zeros(2)
        other_contacts = 0
        wrench = np.zeros(6)
        for index, contact in enumerate(d.contact):
            b1, b2 = m.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.box_id not in (b1, b2):
                continue
            mujoco.mj_contactForce(m, d, index, wrench)
            if wrench[0] <= 1e-6:
                continue
            other = contact.geom1 if b2 == self.box_id else contact.geom2
            if other not in self.palms:
                other_contacts += 1
                continue
            side = self.palms.index(other)
            sign = 1 if b2 == self.box_id else -1
            force = sign * contact.frame.reshape(3, 3).T @ wrench[:3]
            normal[side] += wrench[0]
            upward[side] += force[2]
        rotation = d.xmat[self.base_id].reshape(3, 3)
        box_position = d.qpos[self.box_adr : self.box_adr + 3]
        relative = rotation.T @ (box_position - d.xpos[self.base_id])
        tilt = np.degrees(np.arccos(np.clip(d.xmat[self.box_id].reshape(3, 3)[2, 2], -1, 1)))
        return dict(
            time=d.time,
            payload_weight=self.payload_weight,
            box_height=float(box_position[2]),
            base_height=float(d.qpos[2]),
            relative_position=relative,
            tilt_degrees=float(tilt),
            normal_force=normal,
            upward_force=upward,
            other_contacts=other_contacts,
        )
