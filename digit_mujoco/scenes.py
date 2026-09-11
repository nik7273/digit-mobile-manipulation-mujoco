"""Scene builders; original meshes remain in assets/."""

import mujoco

from .config import PALM
from .paths import HOLD_SCENE, ROOT

PALM_RADIUS = PALM.radius


def hold_model(*, platform=False, scene=HOLD_SCENE):
    """Add approximate rounded pads only to this experiment's robot model.

    Four-dimensional contact includes torsional friction for a soft palm patch;
    no adhesion, weld, or rolling-friction constraint is used to hold the box.
    """
    spec = mujoco.MjSpec.from_file(str(scene))
    add_palms(spec)
    if platform:
        spec.worldbody.add_geom(
            name="pickup-platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.42, 0, 0.875],
            size=[0.20, 0.27, 0.025],
            contype=1,
            conaffinity=15,
            condim=3,
            friction=[0.7, 0.005, 0.0001],
            rgba=[0.5, 0.55, 0.6, 1],
        )
        for y in (-0.21, 0.21):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.52, y, 0.425],
                size=[0.025, 0.025, 0.425],
                contype=1,
                conaffinity=15,
                rgba=[0.3, 0.35, 0.4, 1],
            )
    return spec.compile()


def add_palms(spec):
    """Add the contact pads shared by manipulation experiments."""
    for side in ("left", "right"):
        hand = spec.body(f"{side}-hand")
        hand.add_geom(
            name=f"{side}-palm",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[PALM_RADIUS, 0, 0],
            mass=0.05,
            contype=8,
            conaffinity=4,
            condim=4,
            friction=[PALM.friction, 0.01, 0.0001],
            solref=[0.01, 1],
            rgba=[0.15, 0.35, 0.65, 1],
        )
        hand.add_site(name=f"{side}-palm", size=[0.008, 0, 0], rgba=[1, 0, 0, 1])


def transfer_model():
    # Preserve the original horizontal layout and mesh sizes. Recenter box
    # frames for impedance control, removing mass inferred from visual meshes.
    spec = mujoco.MjSpec.from_file(str(ROOT / "assets/package_scene.xml"))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    for name in ("package", "package2", "package3", "package4", "package5"):
        old = spec.body(name)
        position = old.pos.copy() + [0, 0, 0.1]
        spec.delete(old)
        name = "held-box" if name == "package5" else name
        body = spec.worldbody.add_body(name=name, pos=position)
        body.add_freejoint(name=name)
        body.add_geom(
            name=f"{name}-visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="amazonbody",
            material="amazonbody",
            pos=[0, 0, -0.1],
            quat=[0.5, 0.5, 0.5, 0.5],
            mass=0,
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.15, 0.2, 0.1],
            mass=1,
            contype=4,
            conaffinity=15,
            condim=4,
            friction=[0.7, 0.01, 0.0001],
            solref=[0.01, 1],
            rgba=[0, 0, 0, 0],
        )
    # Ground the original .725 m table and replace its solid collision volume.
    spec.delete(spec.body("table"))
    spec.worldbody.add_geom(
        name="table-visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="tablebody",
        material="tablebody",
        pos=[4, -2, 0.7],
        quat=[0.5, 0.5, 0.5, 0.5],
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_geom(
        name="pickup-platform",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[4, -2, 0.7125],
        size=[0.25, 0.5, 0.0125],
        contype=1,
        conaffinity=15,
        friction=[0.7, 0.005, 0.0001],
        rgba=[0, 0, 0, 0],
    )
    for x in (3.85, 4.15):
        for y in (-2.35, -1.65):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, 0.35],
                size=[0.05, 0.05, 0.35],
                contype=1,
                conaffinity=15,
                rgba=[0, 0, 0, 0],
            )
    add_palms(spec)
    return spec.compile()
