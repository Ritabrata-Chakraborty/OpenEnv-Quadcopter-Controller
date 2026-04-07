# -*- coding: utf-8 -*-
"""Global simulation configuration.

orient:
    "NED" — front-right-down body frame, North-East-Down world frame.
    "ENU" — front-left-up body frame, East-North-Up world frame.

usePrecession:
    Enable gyroscopic precession from rotor inertia in the dynamics.
    Set False when rotor inertia (IRzz) is unknown; effect is negligible.
"""

orient = "NED"
usePrecession = bool(False)
