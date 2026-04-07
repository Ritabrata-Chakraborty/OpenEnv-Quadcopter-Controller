# -*- coding: utf-8 -*-
"""Conversions between world-frame and body-frame velocity/rate representations."""

import numpy as np
from numpy import sin, cos


def phiThetaPsiDotToPQR(phi, theta, psi, phidot, thetadot, psidot):
    """Convert Euler-angle rates (φ̇, θ̇, ψ̇) to body angular rates (p, q, r).

    Parameters
    ----------
    phi, theta, psi         : float  — roll, pitch, yaw (rad)
    phidot, thetadot, psidot: float  — Euler angle rates (rad/s)

    Returns
    -------
    ndarray, shape (3,)  — [p, q, r] in rad/s
    """
    p = -sin(theta)*psidot + phidot
    q =  sin(phi)*cos(theta)*psidot + cos(phi)*thetadot
    r = -sin(phi)*thetadot + cos(phi)*cos(theta)*psidot
    return np.array([p, q, r])


def xyzDotToUVW_euler(phi, theta, psi, xdot, ydot, zdot):
    """Convert world-frame velocity (ẋ, ẏ, ż) to body-frame velocity (u, v, w).

    Parameters
    ----------
    phi, theta, psi : float  — roll, pitch, yaw (rad)
    xdot, ydot, zdot: float  — world-frame velocity components (m/s)

    Returns
    -------
    ndarray, shape (3,)  — [u, v, w] in m/s
    """
    u = xdot*cos(psi)*cos(theta) + ydot*sin(psi)*cos(theta) - zdot*sin(theta)
    v = (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*ydot \
      + (sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))*xdot \
      + zdot*sin(phi)*cos(theta)
    w = (sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))*xdot \
      + (-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*ydot \
      + zdot*cos(phi)*cos(theta)
    return np.array([u, v, w])


def xyzDotToUVW_Flat_euler(phi, theta, psi, xdot, ydot, zdot):
    """Convert world-frame velocity to flat-body velocity (yaw-only rotation, no tilt).

    Parameters
    ----------
    phi, theta, psi : float  — roll, pitch, yaw (rad); phi and theta are ignored
    xdot, ydot, zdot: float  — world-frame velocity components (m/s)

    Returns
    -------
    ndarray, shape (3,)  — [uFlat, vFlat, wFlat] in m/s
    """
    uFlat =  xdot * cos(psi) + ydot * sin(psi)
    vFlat = -xdot * sin(psi) + ydot * cos(psi)
    wFlat = zdot
    return np.array([uFlat, vFlat, wFlat])


def xyzDotToUVW_Flat_quat(q, xdot, ydot, zdot):
    """Convert world-frame velocity to flat-body velocity using quaternion heading.

    Extracts only the yaw component of the quaternion (flat-earth assumption).

    Parameters
    ----------
    q               : array_like, shape (4,)  — [w, x, y, z]
    xdot, ydot, zdot: float  — world-frame velocity components (m/s)

    Returns
    -------
    ndarray, shape (3,)  — [uFlat, vFlat, wFlat] in m/s
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    uFlat =  2*(q0*q3 - q1*q2)*ydot + (q0**2 - q1**2 + q2**2 - q3**2)*xdot
    vFlat = -2*(q0*q3 + q1*q2)*xdot + (q0**2 + q1**2 - q2**2 - q3**2)*ydot
    wFlat = zdot
    return np.array([uFlat, vFlat, wFlat])
