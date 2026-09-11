"""Compatibility entry point; implementation lives in digit_mujoco."""

from digit_mujoco.experiments.pickup import Pickup as Pickup
from digit_mujoco.experiments.pickup import PickupStats as PickupStats
from digit_mujoco.experiments.pickup import main as main
from digit_mujoco.geometry import blend as blend

if __name__ == "__main__":
    main()
