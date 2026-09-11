"""Digit ALIP control in MuJoCo 3. Run --help for viewer/headless options."""
from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from robot_config import INITIAL_QPOS, JOINT_NAMES, LIMITS

ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / 'assets' / 'package_scene.xml'
MODES = {'standing-analytic': 0, 'standing-numeric': 1, 'walking': 2}


def load_controller():
    # Use a distinct module name: never accidentally load the historical Linux .so.
    sys.path.insert(0, str(ROOT / 'build' / 'python'))
    try:
        return importlib.import_module('_digit_alip').Controller
    except ImportError as exc:
        raise RuntimeError('Build the native controller with: python scripts/build_controller.py') from exc


class Simulation:
    def __init__(self, scene=DEFAULT_SCENE, mode='walking', controller=True, goal_x=None, model=None):
        self.model = m = model if model is not None else mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        self.data = d = mujoco.MjData(m)
        self.mode = mode
        if goal_x is not None and (not math.isfinite(goal_x) or not controller):
            raise ValueError('A finite goal requires the ALIP controller')
        self.goal_x = goal_x
        self.controller_type = load_controller() if controller else None
        # Cache indices: XML actuator order is different from the controller order.
        def ids(kind, names):
            result = np.array([mujoco.mj_name2id(m, kind, name) for name in names])
            if np.any(result < 0):
                raise ValueError(f'Model is missing required names: {[n for n, i in zip(names, result) if i < 0]}')
            return result
        self.joint_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES)
        self.actuator_ids = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, JOINT_NAMES[:20])
        self.qadr = m.jnt_qposadr[self.joint_ids]
        self.vadr = m.jnt_dofadr[self.joint_ids]
        self.gear = m.actuator_gear[self.actuator_ids, 0].copy()
        if np.any(self.gear == 0):
            raise ValueError('Motor transmission gear must be nonzero')
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base')
        # The original pose contains rod ball joints as well as motor coordinates.
        if self.base_id < 0 or m.nq < len(INITIAL_QPOS):
            raise ValueError('Scene must contain the Digit v3 robot')
        self.reset()

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:len(INITIAL_QPOS)] = INITIAL_QPOS
        mujoco.mj_normalizeQuat(m, d.qpos)
        mujoco.mj_forward(m, d)
        # Only correct robot/ground penetration; packages must not move the base.
        feet = {m.body(name).id for name in ('left-toe-roll', 'right-toe-roll')}
        correction = 0.0
        for contact in d.contact:
            b1, b2 = m.geom_bodyid[[contact.geom1, contact.geom2]]
            if (b1 == 0 and b2 in feet) or (b2 == 0 and b1 in feet):
                correction = max(correction, -contact.dist * abs(contact.frame[2]))
        d.qpos[2] += correction
        mujoco.mj_forward(m, d)
        self.goal_reached_at = None
        self.controller = (self.controller_type(MODES[self.mode], LIMITS, m.opt.timestep)
                           if self.controller_type else None)

    def observation(self):
        m, d = self.model, self.data
        quat = d.qpos[3:7]
        w, x, y, z = quat
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        c, s = np.cos(yaw), np.sin(yaw)
        # ALIP Extract_Observation_ rotates velocity back using yaw only.
        yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        base = np.concatenate((d.qpos[:7], yaw_rotation.T @ d.qvel[:3], d.qvel[3:6]))
        motors = np.stack((d.qpos[self.qadr[:20]], d.qvel[self.vadr[:20]],
                           d.actuator_force[self.actuator_ids] * self.gear))
        joints = np.stack((d.qpos[self.qadr[20:]], d.qvel[self.vadr[20:]]))
        return base, motors, joints

    def update_goal(self):
        """Upstream forward-walking experiment, driven by simulation time."""
        t = self.data.time
        distance = self.goal_x - self.data.qpos[0]
        if t < 3:
            mode, velocity = 0, 0.0
        else:
            if self.goal_reached_at is None and abs(distance) < .05:
                self.goal_reached_at = t
            if self.goal_reached_at is not None:
                mode = 0 if t - self.goal_reached_at < 10 else 1
                velocity = 0.0
            else:
                mode = 2
                speed = max(.05, .3 * math.sqrt(abs(distance))) if abs(distance) < .6 else .8
                velocity = math.copysign(speed, distance)
        self.controller.set_mode(mode)
        self.controller.set_velocity(velocity, 0.0, 0.0)

    def motor_torques(self):
        """Joint torques in controller order; subclasses can replace arm control."""
        d = self.data
        if self.controller is not None:
            if self.goal_x is not None:
                self.update_goal()
            base, motors, joints = self.observation()
            commands = self.controller.update(d.time, base, motors, joints)
            if not np.isfinite(commands).all():
                raise RuntimeError(f'Nonfinite controller command at t={d.time:.6f}')
            torque = commands[:, 0] + commands[:, 2] * (commands[:, 1] - motors[1])
            return torque
        return np.zeros(20)

    def step(self):
        m, d = self.model, self.data
        torque = self.motor_torques()
        if not np.isfinite(torque).all():
            raise RuntimeError(f'Nonfinite motor torque at t={d.time:.6f}')
        # Retain MuJoCo ctrlrange limits and explicitly bound total joint torque.
        d.ctrl[self.actuator_ids] = np.clip(torque, -LIMITS[0], LIMITS[0]) / self.gear
        previous_time = d.time
        mujoco.mj_step(m, d)
        if d.time <= previous_time or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('MuJoCo state became invalid or reset after instability')
        if np.any(d.warning.number):
            raise RuntimeError(f'MuJoCo reported simulation warnings: {d.warning.number}')


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--mode', choices=MODES, default='walking')
    parser.add_argument('--duration', type=positive, default=60, help='simulation seconds')
    parser.add_argument('--headless', action='store_true', help='run without a window or wall-clock pacing')
    parser.add_argument('--passive', action='store_true', help='physics only, without ALIP (robot will fall)')
    parser.add_argument('--fps', type=positive, default=60, help='viewer refresh rate')
    parser.add_argument('--contacts', action='store_true', help='show contact points')
    parser.add_argument('--goal-x', type=float, help='experimental world-x goal; starts with 3s standing')
    args = parser.parse_args()
    if args.goal_x is not None and (not math.isfinite(args.goal_x) or args.passive):
        parser.error('--goal-x must be finite and cannot be used with --passive')
    sim = Simulation(args.scene, args.mode, controller=not args.passive, goal_x=args.goal_x)
    start = time.perf_counter()
    steps = math.ceil(args.duration / sim.model.opt.timestep)
    if args.headless:
        for _ in range(steps):
            sim.step()
    else:
        from mujoco import viewer as mj_viewer
        # macOS requires the mjpython launcher for the passive viewer.
        with mj_viewer.launch_passive(sim.model, sim.data) as viewer:
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.contacts
            next_frame = 0.0
            for _ in range(steps):
                if not viewer.is_running():
                    break
                sim.step()
                if sim.data.time >= next_frame:
                    viewer.sync()
                    next_frame = sim.data.time + 1 / args.fps
                    remaining = sim.data.time - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)
    elapsed = time.perf_counter() - start
    print(f'MuJoCo {mujoco.__version__}: {sim.data.time:.3f}s simulated in {elapsed:.3f}s '
          f'({sim.data.time / elapsed:.2f}x real time), base height={sim.data.qpos[2]:.3f}m')


if __name__ == '__main__':
    main()
