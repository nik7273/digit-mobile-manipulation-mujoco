"""Common argument validation and manipulation viewer/reporting loop."""

import argparse
import json
import math
import time

import mujoco


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


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
                viewer.cam.lookat[:] = sim.camera_target()
                viewer.cam.distance = getattr(sim, "camera_distance", 2.5)
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
                    if getattr(sim, "follow_camera", False):
                        with viewer.lock():
                            viewer.cam.lookat[:] = sim.camera_target()
                    viewer.sync()
                    next_frame = sim.data.time + 1 / 60
                    time.sleep(max(0, sim.data.time - (time.perf_counter() - start)))
    result = stats.summary()
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def payload_camera_target(sim):
    target = 0.5 * (sim.data.qpos[:3] + sim.data.qpos[sim.box_adr : sim.box_adr + 3])
    target[2] = 0.85
    return target
