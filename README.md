# Digit mobile box manipulation in MuJoCo

Digit v3 walking and box manipulation using an ALIP controller. Runs on MuJoCo 3.13.0.

## Setup

Requires Python 3.12, a C++17 compiler (Xcode Command Line Tools on macOS), and a
checkout of `digit-alip-controller`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/build_controller.py --source /path/to/digit-alip-controller
```

Rebuild after changing Python or controller sources.

## Run

Use `mjpython` on macOS or `python` on Linux. Add `--headless` to run without a
window; use `--help` for options. Durations are in simulation seconds.

```sh
mjpython demo.py                  # Walk through the original scene
mjpython standing_hold.py         # Hold a box while standing
mjpython pickup.py                # Pick up a box from a platform
mjpython carry.py                 # Pick up a box and walk backward
mjpython transfer.py --overview   # Carry the top box from the pile to the table
```

For `transfer.py`, omit `--overview` to follow the robot, or add `--speed-scale 1`
for the original slower pace (default: `2`).

## Test

```sh
python -m pytest -m "not integration"  # Quick checks
python -m pytest                       # Include full simulation runs
ruff check .
ruff format --check .
```

## Videos
### Standing Hold
https://github.com/user-attachments/assets/f1765a7a-1171-4b40-9222-da857656ce0b

### Pickup
https://github.com/user-attachments/assets/2008a632-8d69-43f7-a1d2-bf7fa20c9e27

### Carry
https://github.com/user-attachments/assets/228c4e21-f64a-4632-a967-818e441ec2a8

### Transfer
https://github.com/user-attachments/assets/d94485a6-52eb-4d03-9cdf-3fa7836737a0
