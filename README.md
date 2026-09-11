# Digit mobile manipulation in MuJoCo

Digit v3 simulation with an ALIP walking/standing controller, movable packages,
and a table. Uses MuJoCo 3.13.0 and native C++/pybind11 bindings. A contact-based
standing hold and scripted platform pickup are included. Carrying while walking
is not yet implemented.

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

## Platform pickup

```sh
mjpython pickup.py
python pickup.py --headless --duration 20
```

Starts with the 1 kg box on a fixed platform and arms at rest. The hands approach,
close, confirm sustained contact, then lift and hold. The report verifies that
the box clears the platform and is supported only by the palms. This experiment
uses a known box pose; it does not yet include perception or walking.

## Test

```sh
python -m pytest
```
