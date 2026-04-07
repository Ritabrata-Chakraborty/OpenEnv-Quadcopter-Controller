# -*- coding: utf-8 -*-
"""Quadcopter physical parameters, hover command, and initial state vector.

State vector layout (length 21):
  [0:3]  x, y, z            (m)
  [3:7]  q0, q1, q2, q3     quaternion [w, x, y, z]
  [7:10] xdot, ydot, zdot   (m/s)
  [10:13] p, q, r            body angular rates (rad/s)
  [13]   wM1  (rad/s),  [14] wdotM1  (rad/s²)
  [15]   wM2  (rad/s),  [16] wdotM2  (rad/s²)
  [17]   wM3  (rad/s),  [18] wdotM3  (rad/s²)
  [19]   wM4  (rad/s),  [20] wdotM4  (rad/s²)

Motor numbering: M1 front-left, M2 front-right, M3 rear-right, M4 rear-left
(clockwise from M1).
"""

import numpy as np
from numpy.linalg import inv
import quadnav.sim.controller.utils as utils
import quadnav.sim.controller.config as config


def sys_params():
    """Return a dict of vehicle physical parameters with units.
    
    Returns
    -------
    params : dict
        Vehicle parameters with keys:
        - mB (kg): total mass
        - g (m/s²): gravitational acceleration
        - dxm (m): arm length along x
        - dym (m): arm length along y
        - dzm (m): motor height above CoM
        - IB (kg·m²): inertia tensor
        - IRzz (kg·m²): rotor axial moment of inertia
        - Cd: aerodynamic drag coefficient
        - kTh (N / (rad/s)²): thrust coefficient
        - kTo (N·m / (rad/s)²): torque coefficient
        - minThr (N): minimum total thrust
        - maxThr (N): maximum total thrust
        - minWmotor (rad/s): minimum motor speed
        - maxWmotor (rad/s): maximum motor speed
        - tau (s): motor second-order time constant
        - motorc1 (rad/s per %): speed-to-cmd slope
        - motorc0 (rad/s): speed-to-cmd intercept
        - motordeadband (%): deadband around zero command
    """
    mB   = 1.2
    g    = 9.81
    dxm  = 0.16
    dym  = 0.16
    dzm  = 0.05
    IB   = np.array([[0.0123, 0,      0     ],
                     [0,      0.0123, 0     ],
                     [0,      0,      0.0224]])
    IRzz = 2.7e-5

    params = {}
    params["mB"]   = mB
    params["g"]    = g
    params["dxm"]  = dxm
    params["dym"]  = dym
    params["dzm"]  = dzm
    params["IB"]   = IB
    params["invI"] = inv(IB)
    params["IRzz"] = IRzz
    params["useIntergral"] = bool(False)

    params["Cd"]          = 0.1
    params["kTh"]         = 1.076e-5
    params["kTo"]         = 1.632e-7
    params["mixerFM"]     = makeMixerFM(params)
    params["mixerFMinv"]  = inv(params["mixerFM"])
    params["minThr"]      = 0.1 * 4
    params["maxThr"]      = 9.18 * 4
    params["minWmotor"]   = 75
    params["maxWmotor"]   = 925
    params["tau"]         = 0.015
    params["kp"]          = 1.0
    params["damp"]        = 1.0
    params["motorc1"]     = 8.49
    params["motorc0"]     = 74.7
    params["motordeadband"] = 1

    return params


def makeMixerFM(params):
    """Build the 4×4 force-moment mixer matrix for the given orientation.

    Rows correspond to [F, Mx, My, Mz]; columns to [w1², w2², w3², w4²].
    Motor numbering is M1 front-left, clockwise.
    """
    dxm = params["dxm"]
    dym = params["dym"]
    kTh = params["kTh"]
    kTo = params["kTo"]

    if config.orient == "NED":
        return np.array([[    kTh,      kTh,      kTh,      kTh],
                         [dym*kTh, -dym*kTh, -dym*kTh,  dym*kTh],
                         [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                         [   -kTo,      kTo,     -kTo,      kTo]])
    else:  # ENU
        return np.array([[     kTh,      kTh,      kTh,     kTh],
                         [ dym*kTh, -dym*kTh, -dym*kTh, dym*kTh],
                         [-dxm*kTh, -dxm*kTh,  dxm*kTh, dxm*kTh],
                         [     kTo,     -kTo,      kTo,    -kTo]])


def init_cmd(params):
    """Compute hover equilibrium: [cmd_hover, w_hover, thr_hover, tor_hover].
    
    Returns
    -------
    list
        [cmd_hover (%), w_hover (rad/s), thr_hover (N), tor_hover (N·m)]
    """
    mB  = params["mB"];  g   = params["g"]
    kTh = params["kTh"]; kTo = params["kTo"]
    c1  = params["motorc1"]; c0 = params["motorc0"]

    thr_hover = mB * g / 4.0
    w_hover   = np.sqrt(thr_hover / kTh)
    tor_hover = kTo * w_hover**2
    cmd_hover = (w_hover - c0) / c1
    return [cmd_hover, w_hover, thr_hover, tor_hover]


def init_state(params):
    """Build the 21-element initial state vector for a 1 m hover at the origin.
    
    Returns
    -------
    ndarray (21,)
        State vector: [x, y, z (m), q0-q3, xdot, ydot, zdot (m/s),
        p, q, r (rad/s), wM1-wM4 (rad/s), wdotM1-wdotM4 (rad/s²)]
        NED: z₀ = −1 m (1 m above ground). ENU: z₀ = +1 m.
    """
    x0 = 0.;  y0 = 0.;  z0 = -1.   # m  (NED: negative z is up)
    psi0 = 0.; theta0 = 0.; phi0 = 0.  # rad

    quat = utils.YPRToQuat(psi0, theta0, phi0)
    if config.orient == "ENU":
        z0 = -z0

    s = np.zeros(21)
    s[0]  = x0;      s[1]  = y0;      s[2]  = z0      # position (m)
    s[3]  = quat[0]; s[4]  = quat[1]; s[5]  = quat[2]; s[6] = quat[3]  # quaternion
    # s[7:13] remain 0  (velocities and rates)

    w_hover = params["w_hover"]   # hovering motor speed (rad/s)
    s[13] = w_hover;  s[14] = 0.
    s[15] = w_hover;  s[16] = 0.
    s[17] = w_hover;  s[18] = 0.
    s[19] = w_hover;  s[20] = 0.

    return s
