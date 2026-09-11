"""Compatibility entry point; implementation lives in digit_mujoco."""

from digit_mujoco.experiments.transfer import Transfer as Transfer
from digit_mujoco.experiments.transfer import TransferStats as TransferStats
from digit_mujoco.experiments.transfer import main as main
from digit_mujoco.scenes import transfer_model as transfer_model

if __name__ == "__main__":
    main()
