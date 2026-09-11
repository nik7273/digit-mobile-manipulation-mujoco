import mujoco
import numpy as np
import pytest

from digit_mujoco.experiments.pickup import Pickup, PickupStats, blend
from digit_mujoco.robot import LIMITS


@pytest.mark.integration
def test_pickup_from_platform_to_unsupported_hold():
    sim = Pickup()
    stats = PickupStats()
    m, d = sim.model, sim.data
    assert sim.box_id not in m.eq_obj1id
    assert sim.box_id not in m.eq_obj2id
    assert m.joint("held-box").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.body_mocapid[sim.box_id] == -1
    for i in range(40000):
        sim.step()
        if i % 20 == 19:
            stats.record(sim)
            assert np.all(np.abs(d.ctrl[sim.actuator_ids] * sim.gear) <= LIMITS[0] + 1e-10)
    summary = stats.summary()
    assert summary["passed"], summary
    assert summary["initially_platform_supported"]
    assert summary["max_palm_force_before_closing_n"] == 0
    assert summary["min_platform_clearance_m"] > 0.04
    assert summary["max_platform_force_during_hold_n"] == 0
    assert summary["hold"]["other_box_contacts"] == 0
    assert np.all(d.xfrc_applied == 0)
    assert np.all(d.qfrc_applied == 0)


@pytest.mark.integration
def test_no_squeeze_does_not_lift():
    sim = Pickup(squeeze=0)
    with pytest.raises(RuntimeError, match="Grasp failed"):
        for _ in range(18000):
            sim.step()
    assert "lift" not in [phase for phase, _ in sim.events]
    assert sim.measure()["platform_force_n"] > 0.8 * sim.payload_weight


def test_pickup_reset_and_trajectory_endpoints():
    sim = Pickup()
    initial = sim.data.qpos.copy()
    for _ in range(100):
        sim.step()
    sim.release()
    sim.reset()
    assert sim.phase == "settle"
    assert sim.events == [("settle", 0.0)]
    assert sim.grasp_rotations is None
    assert not sim.released
    np.testing.assert_array_equal(sim.data.qpos, initial)
    start, end = np.array([0, 0, 0]), np.array([1, 2, 3])
    for elapsed, expected in ((0, start), (2, end), (3, end)):
        position, velocity = blend(start, end, elapsed, 2)
        np.testing.assert_allclose(position, expected)
        np.testing.assert_allclose(velocity, 0)
