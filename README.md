# Digit mobile manipulation in MuJoCo

Digit v3 simulation with an ALIP walking/standing controller, movable packages,
and a table. Uses MuJoCo 3.13.0 and native C++/pybind11 bindings. A contact-based
standing hold, pickup, payload walking, and pile-to-table transfer are included.

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

## Walk with a payload

```sh
mjpython carry.py
python carry.py --headless --duration 30
```

Picks up the original textured Amazon box, settles the grasp, then walks backward
away from the table. The camera follows Digit. The box retains its original
.30 × .40 × .20 m dimensions with an explicit 1 kg mass. The original table mesh
is fixed and scaled vertically to a .9 m tabletop for reachability, with separate
top/leg collisions. The original demo assets are unchanged.

Use `--assets simple` for the pickup primitives or `--speed .2` to set the backward
velocity target (maximum .3 m/s). The report checks payload travel, footfalls,
grip slip, and palm-only support. This is a known-pose, straight-line retreat;
it does not yet include turning, navigation, or set-down.

## Original pile to table

```sh
mjpython transfer.py --overview
python transfer.py --headless --duration 100
```

Uses the original five-box pile and table locations, original textures and mesh
sizes, and 1 kg free boxes. The table is grounded and fixed, with separate top
and leg collisions. The scripted route approaches the pile, lifts its top box,
sidesteps clear, and travels to the table for a contact-checked set-down.
Omit `--overview` for a camera that follows Digit. Completion takes about 86
simulation seconds. This uses known object poses and scripted waypoints; the
report checks palm-only carrying and stable table support after release. The current carry has a brief tilt of about 29° while retaining
both palm contacts.

## Test

```sh
python -m pytest
```
