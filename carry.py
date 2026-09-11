"""Compatibility entry point; implementation lives in digit_mujoco."""

from digit_mujoco.experiments.carry import Carry as Carry
from digit_mujoco.experiments.carry import CarryStats as CarryStats
from digit_mujoco.experiments.carry import main as main

if __name__ == "__main__":
    main()
