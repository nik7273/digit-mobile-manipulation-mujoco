"""Build the local ALIP checkout for the active Python interpreter."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "digit-alip-controller")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    venv_cmake = Path(sys.executable).parent / "cmake"
    cmake = str(venv_cmake) if venv_cmake.exists() else shutil.which("cmake")
    if cmake is None:
        parser.error("Install build dependencies: python -m pip install -r requirements-dev.txt")
    build = ROOT / "build" / "controller"
    subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build),
            f"-DDIGIT_CONTROLLER_SOURCE={args.source.expanduser().resolve()}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run([cmake, "--build", str(build), "--parallel", str(args.jobs)], check=True)


if __name__ == "__main__":
    main()
