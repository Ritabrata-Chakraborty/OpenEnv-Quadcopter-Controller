# -*- coding: utf-8 -*-
"""Rotation representations: quaternion ↔ Euler (ZYX), quaternion ↔ DCM, rotation matrix → quaternion.

Quaternion convention: q = [q0, q1, q2, q3] = [w, x, y, z].
Euler convention (ZYX): yaw (ψ), pitch (θ), roll (φ).
"""

import numpy as np
from numpy import sin, cos
from numpy.linalg import norm


def quatToYPR_ZYX(q):
    """Convert unit quaternion to [yaw, pitch, roll] in radians (ZYX convention).

    Parameters
    ----------
    q : array_like, shape (4,)  — [w, x, y, z]

    Returns
    -------
    YPR : ndarray, shape (3,)  — [ψ, θ, φ] in rad
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return threeaxisrot(
        2.0 * (q1*q2 + q0*q3),
        q0**2 + q1**2 - q2**2 - q3**2,
        -2.0 * (q1*q3 - q0*q2),
        2.0 * (q2*q3 + q0*q1),
        q0**2 - q1**2 - q2**2 + q3**2,
    )


def threeaxisrot(r11, r12, r21, r31, r32):
    """Compute ZYX Euler angles from rotation matrix elements.

    Returns [arctan2(r11,r12), arcsin(r21), arctan2(r31,r32)] in rad.
    """
    return np.array([np.arctan2(r11, r12),
                     np.arcsin(r21),
                     np.arctan2(r31, r32)])


def YPRToQuat(r1, r2, r3):
    """Convert ZYX Euler angles to a unit quaternion.

    Parameters
    ----------
    r1 : float  — yaw   ψ (rad)
    r2 : float  — pitch θ (rad)
    r3 : float  — roll  φ (rad)

    Returns
    -------
    q : ndarray, shape (4,)  — [w, x, y, z], normalised
    """
    cr1, sr1 = cos(0.5*r1), sin(0.5*r1)
    cr2, sr2 = cos(0.5*r2), sin(0.5*r2)
    cr3, sr3 = cos(0.5*r3), sin(0.5*r3)

    q0 = cr1*cr2*cr3 + sr1*sr2*sr3
    q1 = cr1*cr2*sr3 - sr1*sr2*cr3
    q2 = cr1*sr2*cr3 + sr1*cr2*sr3
    q3 = sr1*cr2*cr3 - cr1*sr2*sr3

    q = np.array([q0, q1, q2, q3])
    return q / norm(q)


def quat2Dcm(q):
    """Convert unit quaternion to a 3×3 Direction Cosine Matrix (DCM).

    Parameters
    ----------
    q : array_like, shape (4,)  — [w, x, y, z]

    Returns
    -------
    dcm : ndarray, shape (3, 3)
    """
    dcm = np.zeros([3, 3])
    dcm[0, 0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    dcm[0, 1] = 2.0*(q[1]*q[2] - q[0]*q[3])
    dcm[0, 2] = 2.0*(q[1]*q[3] + q[0]*q[2])
    dcm[1, 0] = 2.0*(q[1]*q[2] + q[0]*q[3])
    dcm[1, 1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    dcm[1, 2] = 2.0*(q[2]*q[3] - q[0]*q[1])
    dcm[2, 0] = 2.0*(q[1]*q[3] - q[0]*q[2])
    dcm[2, 1] = 2.0*(q[2]*q[3] + q[0]*q[1])
    dcm[2, 2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    return dcm


def RotToQuat(R):
    """Convert a 3×3 rotation matrix to a unit quaternion (Shepperd's method).

    Parameters
    ----------
    R : ndarray, shape (3, 3)

    Returns
    -------
    q : ndarray, shape (4,)  — [w, x, y, z], normalised, w ≥ 0
    """
    R11, R12, R13 = R[0, 0], R[0, 1], R[0, 2]
    R21, R22, R23 = R[1, 0], R[1, 1], R[1, 2]
    R31, R32, R33 = R[2, 0], R[2, 1], R[2, 2]
    tr = R11 + R22 + R33

    if tr > R11 and tr > R22 and tr > R33:
        e0 = 0.5 * np.sqrt(1 + tr)
        r  = 0.25 / e0
        e1 = (R32 - R23) * r
        e2 = (R13 - R31) * r
        e3 = (R21 - R12) * r
    elif R11 > R22 and R11 > R33:
        e1 = 0.5 * np.sqrt(1 - tr + 2*R11)
        r  = 0.25 / e1
        e0 = (R32 - R23) * r
        e2 = (R12 + R21) * r
        e3 = (R13 + R31) * r
    elif R22 > R33:
        e2 = 0.5 * np.sqrt(1 - tr + 2*R22)
        r  = 0.25 / e2
        e0 = (R13 - R31) * r
        e1 = (R12 + R21) * r
        e3 = (R23 + R32) * r
    else:
        e3 = 0.5 * np.sqrt(1 - tr + 2*R33)
        r  = 0.25 / e3
        e0 = (R21 - R12) * r
        e1 = (R13 + R31) * r
        e2 = (R23 + R32) * r

    q = np.array([e0, e1, e2, e3])
    q = q * np.sign(e0)
    return q / np.sqrt(np.sum(q**2))
