# -*- coding: utf-8 -*-
"""Quaternion algebra: normalization, multiplication, and inversion.

Convention throughout: q = [q0, q1, q2, q3] = [w, x, y, z].
"""

import numpy as np
from numpy.linalg import norm


def vectNormalize(q):
    """Return q divided by its L2 norm (works for any vector or quaternion)."""
    return q / norm(q)


def quatMultiply(q, p):
    """Return the Hamilton product q ⊗ p.

    Parameters
    ----------
    q, p : array_like, shape (4,)
        Unit quaternions [w, x, y, z].
    """
    Q = np.array([[q[0], -q[1], -q[2], -q[3]],
                  [q[1],  q[0], -q[3],  q[2]],
                  [q[2],  q[3],  q[0], -q[1]],
                  [q[3], -q[2],  q[1],  q[0]]])
    return Q @ p


def inverse(q):
    """Return the conjugate of q normalised to unit length (= q^{-1} for unit q)."""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / norm(q)
