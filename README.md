# Digit mobile manipulation in MuJoCo

Digit v3 simulation with an ALIP walking/standing controller, movable packages,
and a table. Uses MuJoCo 3.13.0 and native C++/pybind11 bindings. Autonomous
pick-and-place is not yet implemented.

## Setup

Requires Python 3.12, a C++17 compiler, and a checkout of `digit-alip-controller`.
On macOS, install Xcode Command Line Tools for the compiler.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/build_controller.py --source /path/to/digit-alip-controller
```

Rebuild after changing Python or controller sources. For exact dependency
versions, install `requirements-lock.txt` instead of `requirements-dev.txt`.

## Run

```sh
mjpython demo.py                              # macOS viewer
python demo.py                               # Linux viewer
python demo.py --headless --duration 10       # no window
python demo.py --headless --mode standing-analytic --duration 2
```

Use `--goal-x 1.0` for the experimental walk-to-goal and standing transition.
Walking is the default mode. Use `--help` for all options; `--duration` is in
simulation seconds.

## Test

```sh
python -m pytest
```
