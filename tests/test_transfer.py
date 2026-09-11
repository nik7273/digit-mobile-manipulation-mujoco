import mujoco
import numpy as np
import pytest

from digit_mujoco.experiments.transfer import Transfer, TransferStats
from tests.helpers import mesh_bounds


def test_original_pile_and_unscaled_table():
    sim = Transfer()
    m, d = sim.model, sim.data
    for name, height in [
        ("package3", 0.1),
        ("package2", 0.35),
        ("package", 0.6),
        ("package4", 1.1),
        ("held-box", 1.6),
    ]:
        body = m.body(name).id
        np.testing.assert_allclose(d.xpos[body], [0.45, 0, height])
        assert m.body_mass[body] == pytest.approx(1)
        assert m.joint(name).type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert body not in m.eq_obj1id and body not in m.eq_obj2id
        lower, upper = mesh_bounds(sim, f"{name}-visual")
        np.testing.assert_allclose(upper - lower, [0.3, 0.4, 0.2], atol=1e-6)
    lower, upper = mesh_bounds(sim, "table-visual")
    np.testing.assert_allclose(lower, [3.75, -2.5, 0], atol=1e-6)
    np.testing.assert_allclose(upper, [4.25, -1.5, 0.725], atol=1e-6)
    assert not TransferStats().summary()["passed"]


@pytest.mark.integration
def test_full_contact_only_transfer_and_reset():
    sim = Transfer()
    stats = TransferStats()
    for i in range(200000):
        sim.step()
        if i % 20 == 19:
            stats.record(sim)
        if sim.phase == "done" and sim.data.time - sim.phase_start >= 4:
            break
    result = stats.summary()
    assert result["passed"], result
    assert 20 < result["carry_seconds"] < 35
    assert result["completion_seconds"] < 70
    assert result["palm_only_carry"]
    assert result["final_table_support_n"] == pytest.approx(sim.payload_weight, rel=0.05)
    assert np.all(sim.data.xfrc_applied == 0)
    assert np.all(sim.data.qfrc_applied == 0)
    # The four unpicked boxes remain in the original pile.
    for name, height in [("package3", 0.1), ("package2", 0.3), ("package", 0.5), ("package4", 0.7)]:
        np.testing.assert_allclose(
            sim.data.xpos[sim.model.body(name).id], [0.45, 0, height], atol=0.01
        )
    sim.reset()
    assert sim.data.time == 0 and sim.phase == "settle"
    assert sim.grasp_moment is None and sim.grasp_rotations is None
    np.testing.assert_allclose(sim.data.qpos[sim.box_adr : sim.box_adr + 3], [0.45, 0, 1.6])
    sim.step()


@pytest.mark.parametrize("speed", [0, 0.9, 2.1, float("nan"), float("inf")])
def test_speed_scale_validation(speed):
    with pytest.raises(ValueError, match="Speed scale"):
        Transfer(speed_scale=speed)
