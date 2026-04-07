# -*- coding: utf-8 -*-
"""Default waypoint set used by the trajectory generator.

Returns a fixed 6-waypoint tour used for development and testing.
Replace or override makeWaypoints() to define custom missions.
"""

import numpy as np
from numpy import pi

deg2rad = pi / 180.0


def makeWaypoints():
    """Return (t_wps, wps, y_wps, v_average) for the default mission.

    Returns
    -------
    t_wps : ndarray, shape (N,)
        Arrival times at each waypoint (s).
    wps : ndarray, shape (N, 3)
        Waypoint positions [x, y, z] (m).
    y_wps : ndarray, shape (N,)
        Desired yaw at each waypoint (rad).
    v_average : float
        Average cruise speed used when averVel=1 (m/s).
    """
    v_average = 1.6

    t_ini = 3
    t = np.array([2, 0, 2, 0])

    wp_ini = np.array([0, 0, 0])
    wp = np.array([[ 2,  2,  1],
                   [-2,  3, -3],
                   [-2, -1, -3],
                   [ 3, -2,  1],
                   wp_ini])

    yaw_ini = 0
    yaw = np.array([20, -90, 120, 45])

    t   = np.hstack((t_ini, t)).astype(float)
    wp  = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw_ini, yaw)).astype(float) * deg2rad

    return t, wp, yaw, v_average
