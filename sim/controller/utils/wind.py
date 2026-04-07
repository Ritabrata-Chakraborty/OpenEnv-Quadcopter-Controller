# -*- coding: utf-8 -*-
"""Wind disturbance models for quadcopter simulation.

Supported types (case-insensitive string passed to Wind()):
    'NONE'       — zero wind
    'FIXED'      — constant velocity, heading, and elevation
    'SINE'       — sum-of-sines with fixed median values
    'RANDOMSINE' — sum-of-sines with randomised median values

Wind heading qW1 is measured from the +X axis in the XY plane (rad).
Wind elevation qW2 is positive upward in NED, positive downward in ENU (rad).
"""

import numpy as np
from numpy import sin, pi
import random as rd

deg2rad = pi / 180.0


class Wind:
    """Wind disturbance model.

    Parameters (positional, vary by type)
    -------------------------------------
    NONE       : Wind('None')
    FIXED      : Wind('Fixed', velW, qW1_deg, qW2_deg)
               — Fixed wind: velW (m/s), qW1_deg (deg), qW2_deg (deg)
    SINE       : Wind('Sine',  velW_med, qW1_med_deg, qW2_med_deg)
               — Sine wind: velW_med (m/s), qW1_med_deg (deg), qW2_med_deg (deg)
    RANDOMSINE : Wind('RandomSine', velW_max, velW_min,
                                    qW1_max_deg, qW1_min_deg,
                                    qW2_max_deg, qW2_min_deg)
               — Random sine: velW in (m/s), qW1, qW2 in (deg)
    """

    def __init__(self, *args):
        if len(args) == 0:
            self.windType = 'NONE'
        elif not isinstance(args[0], str):
            raise Exception('First argument must be a wind-type string.')
        else:
            self.windType = args[0].upper()

        if self.windType in ('SINE', 'RANDOMSINE'):
            if self.windType == 'SINE':
                self.velW_med = args[1]
                self.qW1_med  = args[2] * deg2rad
                self.qW2_med  = args[3] * deg2rad
            else:
                velW_max = args[1]; velW_min = args[2]
                qW1_max  = args[3]; qW1_min  = args[4]
                qW2_max  = args[5]; qW2_min  = args[6]
                self.velW_med = (velW_max - velW_min) * rd.random() + velW_min
                self.qW1_med  = ((qW1_max - qW1_min) * rd.random() + qW1_min) * deg2rad
                self.qW2_med  = ((qW2_max - qW2_min) * rd.random() + qW2_min) * deg2rad

            self.velW_a1 = 1.5;  self.velW_f1 = 0.7;  self.velW_d1 = 0.0
            self.velW_a2 = 1.1;  self.velW_f2 = 1.2;  self.velW_d2 = 1.3
            self.velW_a3 = 0.8;  self.velW_f3 = 2.3;  self.velW_d3 = 2.0

            self.qW1_a1 = 15.0 * deg2rad;  self.qW1_f1 = 0.10;  self.qW1_d1 = 0.0
            self.qW1_a2 =  3.0 * deg2rad;  self.qW1_f2 = 0.54;  self.qW1_d2 = 0.0

            self.qW2_a1 = 4.0 * deg2rad;  self.qW2_f1 = 0.10;  self.qW2_d1 = 0.0
            self.qW2_a2 = 0.8 * deg2rad;  self.qW2_f2 = 0.54;  self.qW2_d2 = 0.0

        elif self.windType == 'FIXED':
            self.velW_med = args[1]
            self.qW1_med  = args[2] * deg2rad
            self.qW2_med  = args[3] * deg2rad

        elif self.windType == 'NONE':
            self.velW_med = 0.0
            self.qW1_med  = 0.0
            self.qW2_med  = 0.0

        else:
            raise Exception(f"Unknown wind type '{args[0]}'. "
                            "Choose from: None, Fixed, Sine, RandomSine.")

    def randomWind(self, t):
        """Return (velW, qW1, qW2) for simulation time t (s)."""
        if self.windType in ('SINE', 'RANDOMSINE'):
            velW = (self.velW_a1 * sin(self.velW_f1*t - self.velW_d1)
                  + self.velW_a2 * sin(self.velW_f2*t - self.velW_d2)
                  + self.velW_a3 * sin(self.velW_f3*t - self.velW_d3)
                  + self.velW_med)
            qW1 = (self.qW1_a1 * sin(self.qW1_f1*t - self.qW1_d1)
                 + self.qW1_a2 * sin(self.qW1_f2*t - self.qW1_d2)
                 + self.qW1_med)
            qW2 = (self.qW2_a1 * sin(self.qW2_f1*t - self.qW2_d1)
                 + self.qW2_a2 * sin(self.qW2_f2*t - self.qW2_d2)
                 + self.qW2_med)
            velW = max(0.0, velW)
        else:
            velW = self.velW_med
            qW1  = self.qW1_med
            qW2  = self.qW2_med

        return velW, qW1, qW2
