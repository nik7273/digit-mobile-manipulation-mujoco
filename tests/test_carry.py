import mujoco
import numpy as np
import pytest

from digit_mujoco.experiments.carry import Carry, CarryStats
from digit_mujoco.robot import LIMITS
from tests.helpers import mesh_bounds


@pytest.mark.parametrize("assets", ["simple", "amazon"])
@pytest.mark.integration
def test_pickup_then_walk_with_contact_only_payload(assets):
    sim = Carry(assets=assets)
    m, d = sim.model, sim.data
    assert m.body_mass[sim.box_id] == pytest.approx(1)
    assert sim.box_id not in m.eq_obj1id
    assert sim.box_id not in m.eq_obj2id
    assert m.joint("held-box").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.body_mocapid[sim.box_id] == -1
    stats = CarryStats()
    for i in range(44000):
        sim.step()
        if i % 20 == 19:
            stats.record(sim)
            assert np.all(np.abs(d.ctrl[sim.actuator_ids] * sim.gear) <= LIMITS[0] + 1e-10)
    result = stats.summary()
    assert result["passed"], result
    assert result["payload_travel_m"] > 0.5
    assert min(result["footfalls"]) >= 5
    assert result["other_box_contacts"] == 0
    assert np.all(d.xfrc_applied == 0)
    assert np.all(d.qfrc_applied == 0)
    sim.reset()
    assert sim.phase == "settle"
    assert sim.walk_origin is None and sim.grasp_origin is None
    assert sim.data.time == 0
    sim.step()


def test_original_meshes_match_payload_and_table_collisions():
    sim = Carry()
    lower, upper = mesh_bounds(sim, "amazon-visual")
    center = sim.data.qpos[sim.box_adr : sim.box_adr + 3]
    np.testing.assert_allclose(lower, center - sim.box_halfsize, atol=1e-6)
    np.testing.assert_allclose(upper, center + sim.box_halfsize, atol=1e-6)
    np.testing.assert_allclose(upper - lower, [0.3, 0.4, 0.2], atol=1e-6)
    lower, upper = mesh_bounds(sim, "table-visual")
    assert lower[2] == pytest.approx(0, abs=1e-6)
    assert upper[2] == pytest.approx(sim.platform_top, abs=1e-6)
    assert sim.platform_top == pytest.approx(0.9)


def test_carry_input_validation_and_initial_report():
    for speed in [0, -0.1, 0.31, float("nan")]:
        with pytest.raises(ValueError):
            Carry(speed=speed)
    with pytest.raises(ValueError):
        Carry(assets="missing")
    assert not CarryStats().summary()["passed"]
