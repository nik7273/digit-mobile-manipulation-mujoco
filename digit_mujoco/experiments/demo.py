"""Digit ALIP control in MuJoCo 3. Run --help for viewer/headless options."""

import argparse
import math
import time
from pathlib import Path

import mujoco

from ..paths import DEFAULT_SCENE
from ..runner import positive
from ..simulation import MODES, Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--mode", choices=MODES, default="walking")
    parser.add_argument("--duration", type=positive, default=60, help="simulation seconds")
    parser.add_argument(
        "--headless", action="store_true", help="run without a window or wall-clock pacing"
    )
    parser.add_argument(
        "--passive", action="store_true", help="physics only, without ALIP (robot will fall)"
    )
    parser.add_argument("--fps", type=positive, default=60, help="viewer refresh rate")
    parser.add_argument("--contacts", action="store_true", help="show contact points")
    parser.add_argument(
        "--goal-x", type=float, help="experimental world-x goal; starts with 3s standing"
    )
    args = parser.parse_args()
    if args.goal_x is not None and (not math.isfinite(args.goal_x) or args.passive):
        parser.error("--goal-x must be finite and cannot be used with --passive")
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
    print(
        f"MuJoCo {mujoco.__version__}: {sim.data.time:.3f}s simulated in {elapsed:.3f}s "
        f"({sim.data.time / elapsed:.2f}x real time), base height={sim.data.qpos[2]:.3f}m"
    )


if __name__ == "__main__":
    main()
