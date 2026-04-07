"""Minimal utils for the bundled sim — only what env.py requires.

The original Quadcopter-RL/utils.py also contains training helpers that
depend on PyTorch (replay buffers, soft-update, OU noise, visualisation).
Those are not needed for serving the environment, so this version omits them.
"""

import math
from math import pi
from typing import Any

import numpy as np

from quadnav.sim.parameter import MAX_GOAL_DISTANCE


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# LiDAR binning
# ---------------------------------------------------------------------------

def bin_lidar(distances: np.ndarray, num_bins: int, distance_cap: float) -> np.ndarray:
    """Min-pool raw lidar distances into ``num_bins``, normalized to [0, 1]."""
    n = len(distances)
    bin_size = n / num_bins
    bins = np.ones(num_bins, dtype=np.float32)
    for i in range(num_bins):
        lo = int(i * bin_size)
        hi = int((i + 1) * bin_size)
        bins[i] = np.clip(np.min(distances[lo:hi]) / distance_cap, 0.0, 1.0)
    return bins


# ---------------------------------------------------------------------------
# State computation
# ---------------------------------------------------------------------------

def compute_state(
    lidar_bins: np.ndarray,
    quad: Any,
    goal_pos: np.ndarray,
    prev_action: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Build 45-dim state: [40 lidar bins | goal_dist | goal_angle | prev_vx | prev_vy | prev_w].

    Returns:
        state: np.array of shape ``(45,)``, normalized.
        goal_dist: 2D distance to goal in meters.
    """
    diff_xy = goal_pos - quad.pos[0:2]
    goal_dist = float(np.linalg.norm(diff_xy))
    goal_angle = normalize_angle(np.arctan2(diff_xy[1], diff_xy[0]) - quad.euler[2])

    state = np.concatenate([
        lidar_bins,
        [np.clip(goal_dist / MAX_GOAL_DISTANCE, 0.0, 1.0)],
        [goal_angle / pi],
        [prev_action[0]],
        [prev_action[1]],
        [prev_action[2]],
    ]).astype(np.float32)

    return state, goal_dist
