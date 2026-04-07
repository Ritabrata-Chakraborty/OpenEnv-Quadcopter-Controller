# -*- coding: utf-8 -*-
"""Motor mixer: maps desired total thrust and body moments to per-motor speeds."""

import numpy as np


def mixerFM(quad, thr, moment):
    """Compute motor speed commands from desired thrust and moment vector.

    Solves [F, Mx, My, Mz]^T = mixerFM * [w1^2, w2^2, w3^2, w4^2]^T via the
    pre-inverted mixer matrix, then clamps to valid motor speed range.

    Parameters
    ----------
    quad   : Quadcopter  — provides params["mixerFMinv"], ["minWmotor"], ["maxWmotor"]
    thr    : float       — desired total thrust (N)
    moment : array_like, shape (3,)  — desired [Mx, My, Mz] (N·m)

    Returns
    -------
    w_cmd : ndarray, shape (4,)  — motor speed commands (rad/s)
    """
    t = np.array([thr, moment[0], moment[1], moment[2]])
    w_cmd = np.sqrt(np.clip(
        np.dot(quad.params["mixerFMinv"], t),
        quad.params["minWmotor"]**2,
        quad.params["maxWmotor"]**2,
    ))
    return w_cmd
