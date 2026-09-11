"""Compatibility entry point; implementation lives in digit_mujoco."""

from digit_mujoco.experiments.demo import main as main
from digit_mujoco.paths import DEFAULT_SCENE as DEFAULT_SCENE
from digit_mujoco.paths import ROOT as ROOT
from digit_mujoco.runner import positive as positive
from digit_mujoco.simulation import MODES as MODES
from digit_mujoco.simulation import Simulation as Simulation
from digit_mujoco.simulation import load_controller as load_controller

if __name__ == "__main__":
    main()
