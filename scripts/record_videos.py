"""Record the README tasks as real-time H.264 MP4s. Requires FFmpeg on PATH."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

# Keep the recorder usable without installing the checkout as a package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from digit_mujoco.experiments.carry import Carry  # noqa: E402
from digit_mujoco.experiments.hold import StandingHold  # noqa: E402
from digit_mujoco.experiments.pickup import Pickup  # noqa: E402
from digit_mujoco.experiments.transfer import Transfer  # noqa: E402
from digit_mujoco.metrics import CarryStats, HoldStats, PickupStats, TransferStats  # noqa: E402
from digit_mujoco.simulation import Simulation  # noqa: E402

TASKS = {
    "demo": (Simulation, None, 10),
    "standing_hold": (StandingHold, HoldStats, 12),
    "pickup": (Pickup, PickupStats, 16),
    "carry": (Carry, CarryStats, 30),
    "transfer": (Transfer, TransferStats, 90),
}


def camera_for(task, sim):
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -20
    camera.distance = 2.8
    if task == "transfer":
        camera.lookat[:] = [1.55, -0.9, 0.8]
        camera.distance = 5.8
        camera.azimuth = 90
        camera.elevation = -22
    elif task == "demo":
        camera.lookat[:] = sim.data.qpos[:3]
        camera.lookat[2] = 0.85
    else:
        camera.lookat[:] = sim.camera_target()
    return camera


def record(task, args, ffmpeg):
    constructor, stats_type, duration = TASKS[task]
    sim = constructor()
    stats = stats_type() if stats_type else None
    sim.model.vis.global_.offwidth = args.width
    sim.model.vis.global_.offheight = args.height
    camera = camera_for(task, sim)
    output = args.output / f"{task}.mp4"
    partial = args.output / f".{task}.partial.mp4"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite to replace it")
    if args.preview:
        for _ in range(round(1 / sim.model.opt.timestep)):
            sim.step()
        with mujoco.Renderer(sim.model, args.height, args.width) as renderer:
            renderer.update_scene(sim.data, camera=camera)
            preview = args.output / f"{task}.png"
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{args.width}x{args.height}",
                    "-i",
                    "pipe:0",
                    "-frames:v",
                    "1",
                    str(preview),
                ],
                input=renderer.render().tobytes(),
                check=True,
            )
        print(preview, flush=True)
        return

    process = None
    steps = 0
    frames = 0
    next_progress = 0
    min_base_height = float("inf")
    end_time = duration
    try:
        with mujoco.Renderer(sim.model, args.height, args.width) as renderer:
            process = subprocess.Popen(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{args.width}x{args.height}",
                    "-framerate",
                    str(args.fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-threads",
                    "2",
                    str(partial),
                ],
                stdin=subprocess.PIPE,
            )
            while frames < math.ceil(end_time * args.fps):
                # Sample evenly in simulation time; rendering/encoding speed cannot
                # change the controller timestep or the video's playback speed.
                sample_time = frames / args.fps
                while sim.data.time + 1e-9 < sample_time:
                    sim.step()
                    steps += 1
                    min_base_height = min(min_base_height, float(sim.data.qpos[2]))
                    if stats is not None and steps % 20 == 0:
                        stats.record(sim)
                    if task == "transfer" and sim.phase == "done":
                        end_time = min(duration, sim.phase_start + 4)
                if task == "carry":
                    camera.lookat[:] = sim.camera_target()
                renderer.update_scene(sim.data, camera=camera)
                process.stdin.write(renderer.render().tobytes())
                frames += 1
                if sim.data.time >= next_progress:
                    print(
                        f"{task}: {sim.data.time:.1f}s {getattr(sim, 'phase', '')}",
                        flush=True,
                    )
                    next_progress += 10
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg failed while recording {task}")
        result = stats.summary() if stats else {"passed": min_base_height > 0.8}
        if not result["passed"]:
            raise RuntimeError(f"{task} validation failed: {result}")
        partial.replace(output)
        print(
            json.dumps(
                {
                    "video": str(output),
                    "frames": frames,
                    "seconds": frames / args.fps,
                    "bytes": output.stat().st_size,
                    "validation": result,
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if partial.exists():
            partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=ROOT / "videos")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preview", action="store_true", help="save one PNG per task for camera checks"
    )
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2 or args.fps <= 0:
        parser.error("width/height must be positive even numbers and fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg must be installed and available on PATH")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        record(task, args, ffmpeg)


if __name__ == "__main__":
    main()
