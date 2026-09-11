"""Contact-only, two-arm standing hold experiment."""
from __future__ import annotations

import argparse
import json
import math
import time

import mujoco
import numpy as np

from demo import ROOT, Simulation, positive

SCENE = ROOT / 'assets' / 'standing_hold.xml'
PALM_RADIUS = .035


def hold_model(*, platform=False):
    """Add approximate rounded pads only to this experiment's robot model.

    Four-dimensional contact includes torsional friction for a soft palm patch;
    no adhesion, weld, or rolling-friction constraint is used to hold the box.
    """
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for side in ('left', 'right'):
        hand = spec.body(f'{side}-hand')
        hand.add_geom(name=f'{side}-palm', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[PALM_RADIUS, 0, 0], mass=.05, contype=8, conaffinity=4,
                      condim=4, friction=[.7, .01, .0001], solref=[.01, 1],
                      rgba=[.15, .35, .65, 1])
        hand.add_site(name=f'{side}-palm', size=[.008, 0, 0], rgba=[1, 0, 0, 1])
    if platform:
        spec.worldbody.add_geom(
            name='pickup-platform', type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[.42, 0, .875], size=[.20, .27, .025],
            contype=1, conaffinity=15, condim=3, friction=[.7, .005, .0001],
            rgba=[.5, .55, .6, 1])
        for y in (-.21, .21):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, pos=[.52, y, .425],
                size=[.025, .025, .425], contype=1, conaffinity=15,
                rgba=[.3, .35, .4, 1])
    return spec.compile()


class StandingHold(Simulation):
    def __init__(self, squeeze=20.0, *, model=None):
        if not math.isfinite(squeeze) or squeeze < 0:
            raise ValueError('Squeeze force must be finite and nonnegative')
        self.squeeze = squeeze
        self.position_gain = 800
        self.velocity_gain = 40
        super().__init__(model=hold_model() if model is None else model, mode='standing-analytic')

    def reset(self):
        super().reset()
        m, d = self.model, self.data
        self.released = False
        # Start in a grasp, not a pickup. The box is free from the first step.
        d.qpos[0:2] = 0
        mujoco.mj_forward(m, d)
        self.sites = [m.site(f'{side}-palm').id for side in ('left', 'right')]
        self.palms = [m.geom(f'{side}-palm').id for side in ('left', 'right')]
        self.box_id = m.body('held-box').id
        self.payload_weight = m.body_mass[self.box_id] * abs(m.opt.gravity[2])
        self.box_adr = m.joint('held-box').qposadr[0]
        self.center_offset = np.array([.35, 0, .05])
        half_width = m.geom_size[m.geom('held-box').id, 1]
        self.offsets = [self.center_offset + [0, sign*(half_width+PALM_RADIUS), 0]
                        for sign in (1, -1)]
        self.jac = np.zeros((3, m.nv))
        self.base_jac = np.zeros((3, m.nv))
        for side, site in enumerate(self.sites):
            idx = slice(12+4*side, 16+4*side)
            target = d.xpos[self.base_id] + d.xmat[self.base_id].reshape(3,3) @ self.offsets[side]
            for _ in range(150):
                mujoco.mj_forward(m, d)
                error = target - d.site_xpos[site]
                if np.linalg.norm(error) < 1e-5:
                    break
                mujoco.mj_jacSite(m, d, self.jac, None, site)
                jac = self.jac[:, self.vadr[idx]]
                delta = jac.T @ np.linalg.solve(jac @ jac.T + .001*np.eye(3), error)
                d.qpos[self.qadr[idx]] += np.clip(delta, -.1, .1)
                limits = m.jnt_range[self.joint_ids[idx]]
                d.qpos[self.qadr[idx]] = np.clip(d.qpos[self.qadr[idx]], limits[:,0], limits[:,1])
            mujoco.mj_forward(m, d)
            error = target - d.site_xpos[site]
            if np.linalg.norm(error) > .005:
                raise RuntimeError(f'Initial palm target unreachable: {error}')
        rotation = d.xmat[self.base_id].reshape(3,3)
        d.qpos[self.box_adr:self.box_adr+3] = d.xpos[self.base_id] + rotation @ self.center_offset
        d.qpos[self.box_adr+3:self.box_adr+7] = d.qpos[3:7]
        self.posture = d.qpos[self.qadr[12:20]].copy()
        mujoco.mj_forward(m, d)

    def release(self):
        """Open the palms without changing the box state or applying box forces."""
        self.released = True

    def palm_command(self, side):
        """Desired world position/velocity and feedforward force for one palm."""
        m, d = self.model, self.data
        rotation = d.xmat[self.base_id].reshape(3,3)
        offset = self.offsets[side].copy()
        if self.released:
            offset[1] += .12 if side == 0 else -.12
        target = d.xpos[self.base_id] + rotation @ offset
        mujoco.mj_jac(m, d, self.base_jac, None, target, self.base_id)
        velocity = self.base_jac @ d.qvel
        force = np.zeros(3)
        if not self.released:
            force += rotation @ np.array([0, -self.squeeze if side == 0 else self.squeeze, 0])
            force += np.array([0, 0, self.payload_weight/2])
        return target, velocity, force

    def motor_torques(self):
        torque = super().motor_torques()
        m, d = self.model, self.data
        for side, site in enumerate(self.sites):
            idx = slice(12+4*side, 16+4*side)
            dofs = self.vadr[idx]
            target, velocity, feedforward = self.palm_command(side)
            mujoco.mj_jacSite(m, d, self.jac, None, site)
            # Damping uses motion relative to the desired trajectory.
            velocity_error = velocity - self.jac @ d.qvel
            force = self.position_gain*(target-d.site_xpos[site]) + self.velocity_gain*velocity_error + feedforward
            jac = self.jac[:, dofs]
            # The fourth arm DOF softly prefers the initialized posture.
            null = np.eye(4) - np.linalg.pinv(jac) @ jac
            posture = 10*(self.posture[side*4:side*4+4]-d.qpos[self.qadr[idx]]) - 2*d.qvel[dofs]
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
        box_position = d.qpos[self.box_adr:self.box_adr+3]
        relative = rotation.T @ (box_position - d.xpos[self.base_id])
        tilt = np.degrees(np.arccos(np.clip(d.xmat[self.box_id].reshape(3,3)[2,2], -1, 1)))
        return dict(time=d.time, payload_weight=self.payload_weight, box_height=float(box_position[2]),
                    base_height=float(d.qpos[2]), relative_position=relative,
                    tilt_degrees=float(tilt), normal_force=normal,
                    upward_force=upward, other_contacts=other_contacts)


class HoldStats:
    """Sample at 100 Hz after a two-second settling period."""
    def __init__(self):
        self.samples = []

    def record(self, sim):
        if sim.data.time >= 2:
            self.samples.append(sim.measure())

    def summary(self):
        if not self.samples:
            return {'passed': False, 'reason': 'Need samples after the 2s settling period'}
        samples = self.samples
        positions = np.array([s['relative_position'] for s in samples])
        drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
        normal = np.array([s['normal_force'] for s in samples])
        upward = np.array([s['upward_force'] for s in samples])
        bilateral = float(np.mean(np.all(normal > 1, axis=1)))
        min_height = min(s['box_height'] for s in samples)
        min_base = min(s['base_height'] for s in samples)
        max_base = max(s['base_height'] for s in samples)
        tilt = max(s['tilt_degrees'] for s in samples)
        other = sum(s['other_contacts'] for s in samples)
        support = float(np.mean(upward.sum(axis=1)))
        weight = samples[0]['payload_weight']
        passed = (min_height > .8 and min_base > .8 and max_base < 1.3 and drift < .02
                  and tilt < 15 and bilateral > .99 and other == 0
                  and abs(support - weight) < .2*weight)
        return dict(passed=bool(passed), measured_seconds=float(samples[-1]['time']-samples[0]['time']),
                    min_box_height_m=min_height, max_relative_drift_m=drift,
                    max_box_tilt_degrees=tilt, bilateral_contact_fraction=bilateral,
                    mean_palm_normal_force_n=normal.mean(axis=0).tolist(),
                    mean_palm_upward_force_n=upward.mean(axis=0).tolist(),
                    other_box_contacts=other)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=positive, default=20, help='simulation seconds, at least 3')
    parser.add_argument('--squeeze', type=positive, default=20, help='inward force per palm in newtons')
    parser.add_argument('--contacts', action='store_true')
    args = parser.parse_args()
    if args.duration < 3:
        parser.error('--duration must be at least 3 seconds to measure the settled hold')
    run_experiment(StandingHold(squeeze=args.squeeze), HoldStats(), args)


def run_experiment(sim, stats, args):
    """Run a manipulation experiment with common viewer pacing and reporting."""
    steps = math.ceil(args.duration / sim.model.opt.timestep)
    def step(index):
        sim.step()
        if index % 20 == 19:
            stats.record(sim)
    if args.headless:
        for i in range(steps):
            step(i)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(sim.model, sim.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [.15, 0, .9]
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.contacts
            start = time.perf_counter()
            next_frame = 0.0
            for i in range(steps):
                if not viewer.is_running():
                    break
                step(i)
                if sim.data.time >= next_frame:
                    viewer.sync()
                    next_frame = sim.data.time + 1/60
                    time.sleep(max(0, sim.data.time - (time.perf_counter() - start)))
    result = stats.summary()
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
