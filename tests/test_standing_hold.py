import mujoco
import numpy as np
import pytest

from robot_config import LIMITS
from standing_hold import HoldStats, StandingHold


def test_contact_only_hold_then_release():
    sim = StandingHold()
    m, d = sim.model, sim.data
    assert m.body_mass[sim.box_id] == pytest.approx(1.0)
    assert m.joint('held-box').type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.body_mocapid[sim.box_id] == -1
    assert sim.box_id not in m.eq_obj1id
    assert sim.box_id not in m.eq_obj2id
    stats = HoldStats()
    for i in range(40000):  # 20 seconds, including the settling transient
        sim.step()
        if i % 20 == 19:
            stats.record(sim)
            assert np.all(np.abs(d.ctrl[sim.actuator_ids] * sim.gear) <= LIMITS[0] + 1e-10)
    result = stats.summary()
    assert result['passed'], result
    assert result['measured_seconds'] > 17
    assert result['other_box_contacts'] == 0
    # The only forces supporting the free payload must be palm contact forces.
    assert np.all(d.xfrc_applied == 0)
    assert np.all(d.qfrc_applied == 0)
    before = d.qpos.copy()
    height = sim.measure()['box_height']
    sim.release()
    np.testing.assert_array_equal(d.qpos, before)
    for _ in range(4000):
        sim.step()
    released = sim.measure()
    assert released['box_height'] < height - .5
    assert np.all(released['normal_force'] < 1)


def test_hold_reset_and_invalid_squeeze():
    with pytest.raises(ValueError):
        StandingHold(squeeze=-1)
    with pytest.raises(ValueError):
        StandingHold(squeeze=float('nan'))
    sim = StandingHold()
    initial = sim.data.qpos.copy()
    sim.release()
    sim.step()
    sim.reset()
    assert not sim.released
    assert sim.data.time == 0
    np.testing.assert_array_equal(sim.data.qpos, initial)
    sim.step()
