"""Small trajectory and frame transforms shared by controllers."""

import numpy as np


def blend(start, end, elapsed, duration):
    """Quintic position interpolation with zero endpoint velocity/acceleration."""
    u = np.clip(elapsed / duration, 0, 1)
    a = 10 * u**3 - 15 * u**4 + 6 * u**5
    da = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    return start + a * (end - start), da * (end - start)


def yaw_rotation(quaternion):
    w, x, y, z = quaternion
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
