import gc
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from digit_mujoco.robot import LIMITS
from digit_mujoco.simulation import ROOT, Simulation, load_controller


def test_scene_and_reset_preserve_objects():
    sim = Simulation(controller=False)
    assert sim.model.nu == 20
    assert sim.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    package = sim.model.joint("package").qposadr[0]
    np.testing.assert_array_equal(
        sim.data.qpos[package : package + 7], sim.model.qpos0[package : package + 7]
    )
    sim.step()
    sim.reset()
    assert sim.data.time == 0
    assert np.isfinite(sim.data.qpos).all()


def test_mapping_includes_arms_and_actual_torques():
    sim = Simulation(controller=False)
    d, m = sim.data, sim.model
    d.ctrl[sim.actuator_ids] = np.linspace(-0.1, 0.1, 20)
    mujoco.mj_forward(m, d)
    _, motors, joints = sim.observation()
    assert motors.shape == (3, 20) and joints.shape == (2, 10)
    assert np.any(motors[0, 12:] != 0)
    for i, act in enumerate(sim.actuator_ids):
        joint = m.actuator_trnid[act, 0]
        assert sim.qadr[i] == m.jnt_qposadr[joint]
        assert motors[2, i] == pytest.approx(d.actuator_force[act] * m.actuator_gear[act, 0])


def test_velocity_convention_matches_alip():
    sim = Simulation(controller=False)
    # Include pitch/roll: a full body rotation would violate ALIP's yaw-only inverse.
    quat = np.empty(4)
    mujoco.mju_euler2Quat(quat, np.array([0.2, -0.3, 0.7]), "xyz")
    sim.data.qpos[3:7] = quat
    sim.data.qvel[:3] = [1, 2, 3]
    base, _, _ = sim.observation()
    w, x, y, z = quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    np.testing.assert_allclose(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ base[7:10], [1, 2, 3])


def test_binding_rejects_bad_inputs_and_owns_output():
    sim = Simulation(controller=False)
    cls = load_controller()
    ctrl = cls(2, LIMITS, sim.model.opt.timestep)
    base, motors, joints = sim.observation()
    with pytest.raises(ValueError):
        ctrl.update(0, base[:12], motors, joints)
    with pytest.raises(ValueError):
        ctrl.update(0, base, np.full((3, 20), np.nan), joints)
    command = ctrl.update(0, base, motors, joints)
    saved = command.copy()
    with pytest.raises(ValueError):
        ctrl.update(0, base, motors, joints)
    del ctrl
    gc.collect()
    np.testing.assert_array_equal(command, saved)
    assert command.flags.owndata and np.isfinite(command).all()


def test_closed_loop_and_controller_reset():
    sim = Simulation()
    initial = sim.data.qpos.copy()
    for _ in range(2000):
        sim.step()
    assert sim.data.time == pytest.approx(1)
    assert 0.7 < sim.data.qpos[2] < 1.3
    assert not np.any(sim.data.warning.number)
    assert np.any(sim.data.ctrl[sim.actuator_ids[12:]] != 0)
    assert np.all(np.abs(sim.data.ctrl[sim.actuator_ids] * sim.gear) <= LIMITS[0] + 1e-10)
    sim.reset()
    np.testing.assert_array_equal(sim.data.qpos, initial)
    sim.step()


def test_cli_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "demo.py"), "--headless", "--duration", ".005"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MuJoCo 3.13.0" in result.stdout


def test_goal_arrival_stays_stopped_and_reset_restarts_experiment():
    sim = Simulation(goal_x=1.0)
    sim.data.time = 3.0
    sim.data.qpos[0] = 1.0
    sim.update_goal()
    assert sim.goal_reached_at == 3.0
    # Small drift must not restart walking after arrival.
    sim.data.qpos[0] = 1.1
    sim.data.time = 14.0
    sim.update_goal()
    assert sim.goal_reached_at == 3.0
    sim.reset()
    assert sim.goal_reached_at is None
    assert sim.data.qpos[0] == pytest.approx(-1.003237)
    sim.step()


def test_goal_and_native_target_validation():
    with pytest.raises(ValueError):
        Simulation(goal_x=float("nan"))
    with pytest.raises(ValueError):
        Simulation(controller=False, goal_x=1.0)
    sim = Simulation()
    with pytest.raises(ValueError):
        sim.controller.set_mode(3)
    with pytest.raises(ValueError):
        sim.controller.set_velocity(float("nan"), 0, 0)
