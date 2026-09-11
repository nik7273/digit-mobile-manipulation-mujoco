"""Contact-only, two-arm standing hold experiment."""

import argparse

import mujoco
import numpy as np

from ..config import PALM
from ..manipulation import PalmImpedance
from ..metrics import HoldStats as HoldStats
from ..runner import positive, run_experiment
from ..scenes import hold_model


class StandingHold(PalmImpedance):
    def __init__(self, squeeze=PALM.squeeze, *, model=None):
        super().__init__(squeeze=squeeze, model=hold_model() if model is None else model)

    def reset(self):
        super().reset()
        m, d = self.model, self.data
        # This experiment alone starts with the box already between the palms.
        for side, site in enumerate(self.sites):
            idx = slice(12 + 4 * side, 16 + 4 * side)
            target = d.xpos[self.base_id] + d.xmat[self.base_id].reshape(3, 3) @ self.offsets[side]
            for _ in range(150):
                mujoco.mj_forward(m, d)
                error = target - d.site_xpos[site]
                if np.linalg.norm(error) < 1e-5:
                    break
                mujoco.mj_jacSite(m, d, self.jac, None, site)
                jac = self.jac[:, self.vadr[idx]]
                delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.001 * np.eye(3), error)
                d.qpos[self.qadr[idx]] += np.clip(delta, -0.1, 0.1)
                limits = m.jnt_range[self.joint_ids[idx]]
                d.qpos[self.qadr[idx]] = np.clip(d.qpos[self.qadr[idx]], limits[:, 0], limits[:, 1])
            mujoco.mj_forward(m, d)
            error = target - d.site_xpos[site]
            if np.linalg.norm(error) > 0.005:
                raise RuntimeError(f"Initial palm target unreachable: {error}")
        rotation = d.xmat[self.base_id].reshape(3, 3)
        d.qpos[self.box_adr : self.box_adr + 3] = (
            d.xpos[self.base_id] + rotation @ self.center_offset
        )
        d.qpos[self.box_adr + 3 : self.box_adr + 7] = d.qpos[3:7]
        self.posture = d.qpos[self.qadr[12:20]].copy()
        mujoco.mj_forward(m, d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration", type=positive, default=20, help="simulation seconds, at least 3"
    )
    parser.add_argument(
        "--squeeze", type=positive, default=PALM.squeeze, help="inward force per palm in newtons"
    )
    parser.add_argument("--contacts", action="store_true")
    args = parser.parse_args()
    if args.duration < 3:
        parser.error("--duration must be at least 3 seconds to measure the settled hold")
    run_experiment(StandingHold(squeeze=args.squeeze), HoldStats(), args)


if __name__ == "__main__":
    main()
