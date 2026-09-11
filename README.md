# Digit mobile manipulation in MuJoCo

Digit v3 simulation with an ALIP walking/standing controller, movable packages,
and a table. Uses MuJoCo 3.13.0 and native C++/pybind11 bindings. A contact-based
standing hold is included; autonomous pickup and carrying while walking are not
yet implemented.

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

## Standing hold

```sh
mjpython standing_hold.py                       # macOS viewer
python standing_hold.py --headless --duration 30
```

Starts with a free 1 kg box between rounded palm pads. Cartesian arm impedance
and inward squeeze support it while ALIP controls standing. There is no box
attachment or external support. The report checks slip, tilt, bilateral contact,
and the weight supported by the palms after 2 seconds of settling.
Use `--squeeze` to tune inward force per hand (default 20 N).

## Test

```sh
python -m pytest
```
