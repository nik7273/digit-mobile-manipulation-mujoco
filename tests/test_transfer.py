import mujoco
import numpy as np
import pytest

from transfer import Transfer, TransferStats
from test_carry import mesh_bounds


def test_original_pile_and_unscaled_table():
    sim = Transfer()
    m, d = sim.model, sim.data
    for name, height in [('package3', .1), ('package2', .35), ('package', .6),
                         ('package4', 1.1), ('held-box', 1.6)]:
        body = m.body(name).id
        np.testing.assert_allclose(d.xpos[body], [.45, 0, height])
        assert m.body_mass[body] == pytest.approx(1)
        assert m.joint(name).type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert body not in m.eq_obj1id and body not in m.eq_obj2id
        lower, upper = mesh_bounds(sim, f'{name}-visual')
        np.testing.assert_allclose(upper-lower, [.3,.4,.2], atol=1e-6)
    lower, upper = mesh_bounds(sim, 'table-visual')
    np.testing.assert_allclose(lower, [3.75,-2.5,0], atol=1e-6)
    np.testing.assert_allclose(upper, [4.25,-1.5,.725], atol=1e-6)
    assert not TransferStats().summary()['passed']


def test_full_contact_only_transfer_and_reset():
    sim = Transfer()
    stats = TransferStats()
    for i in range(200000):
        sim.step()
        if i % 20 == 19:
            stats.record(sim)
        if sim.phase == 'done' and sim.data.time-sim.phase_start >= 4:
            break
    result = stats.summary()
    assert result['passed'], result
    assert result['carry_seconds'] > 40
    assert result['palm_only_carry']
    # The lower pile pickup has a transient pitch excursion during gait entry.
    assert result['max_carry_tilt_degrees'] < 35
    assert result['final_table_support_n'] == pytest.approx(sim.payload_weight, rel=.05)
    assert np.all(sim.data.xfrc_applied == 0)
    assert np.all(sim.data.qfrc_applied == 0)
    # The four unpicked boxes remain in the original pile.
    for name, height in [('package3', .1), ('package2', .3), ('package', .5), ('package4', .7)]:
        np.testing.assert_allclose(sim.data.xpos[sim.model.body(name).id], [.45,0,height], atol=.01)
    sim.reset()
    assert sim.data.time == 0 and sim.phase == 'settle'
    assert sim.grasp_moment is None and sim.grasp_rotations is None
    np.testing.assert_allclose(sim.data.qpos[sim.box_adr:sim.box_adr+3], [.45,0,1.6])
    sim.step()
