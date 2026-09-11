"""Compatibility entry point; implementation lives in digit_mujoco."""

from digit_mujoco.experiments.hold import HoldStats as HoldStats
from digit_mujoco.experiments.hold import StandingHold as StandingHold
from digit_mujoco.experiments.hold import main as main
from digit_mujoco.runner import run_experiment as run_experiment
from digit_mujoco.scenes import PALM_RADIUS as PALM_RADIUS
from digit_mujoco.scenes import add_palms as add_palms
from digit_mujoco.scenes import hold_model as hold_model

if __name__ == "__main__":
    main()
