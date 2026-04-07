# -*- coding: utf-8 -*-
"""Quadcopter rigid-body dynamics with second-order motor model.

Integrates the 21-state ODE using the Dormand-Prince (dopri5) solver.
State vector layout — see vehicle/params.py.
"""

import numpy as np
from numpy import sin, cos, sign
from scipy.integrate import ode

from quadnav.sim.controller.vehicle.params import sys_params, init_cmd, init_state
import quadnav.sim.controller.utils as utils
import quadnav.sim.controller.config as config


class Quadcopter:
    """Full 6-DOF quadcopter simulation with second-order motor dynamics.

    Parameters
    ----------
    Ti : float
        Initial simulation time (s)
    """

    def __init__(self, Ti):

        self.params = sys_params()
        ini_hover = init_cmd(self.params)
        self.params["FF"]        = ini_hover[0]
        self.params["w_hover"]   = ini_hover[1]
        self.params["thr_hover"] = ini_hover[2]
        self.thr = np.ones(4) * ini_hover[2]
        self.tor = np.ones(4) * ini_hover[3]

        self.state  = init_state(self.params)
        self.pos    = self.state[0:3]
        self.quat   = self.state[3:7]
        self.vel    = self.state[7:10]
        self.omega  = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15],
                                self.state[17], self.state[19]])
        self.vel_dot   = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc       = np.zeros(3)

        self.extended_state()
        self.forces()

        self.integrator = ode(self.state_dot).set_integrator(
            'dopri5', first_step=0.00005, atol=1e-5, rtol=1e-5,
        )
        self.integrator.set_initial_value(self.state, Ti)

    def extended_state(self):
        """Compute DCM and Euler angles from the current quaternion."""
        self.dcm   = utils.quat2Dcm(self.quat)
        YPR        = utils.quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1]  # reorder to [φ, θ, ψ]
        self.psi   = YPR[0]
        self.theta = YPR[1]
        self.phi   = YPR[2]

    def forces(self):
        """Update per-motor thrust and torque from current motor speeds."""
        self.thr = self.params["kTh"] * self.wMotor**2
        self.tor = self.params["kTo"] * self.wMotor**2

    def state_dot(self, t, state, cmd, wind):
        """ODE right-hand side: returns d(state)/dt.

        Parameters
        ----------
        t : float
            Current time (s)
        state : ndarray (21,)
            Current state vector [x, y, z, q0-q3, xdot, ydot, zdot,
            p, q, r, wM1-M4, wdotM1-M4] in appropriate units
        cmd : ndarray (4,)
            Motor speed commands (rad/s)
        wind : Wind
            Wind disturbance model
        """
        mB   = self.params["mB"];   g    = self.params["g"]
        dxm  = self.params["dxm"];  dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0, 0]; IByy = IB[1, 1]; IBzz = IB[2, 2]
        Cd   = self.params["Cd"]
        kTh  = self.params["kTh"]; kTo  = self.params["kTo"]
        tau  = self.params["tau"]; kp   = self.params["kp"]; damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]
        IRzz = self.params["IRzz"]
        uP   = 1 if config.usePrecession else 0

        x  = state[0];  y  = state[1];  z  = state[2]
        q0 = state[3];  q1 = state[4];  q2 = state[5];  q3 = state[6]
        xdot = state[7]; ydot = state[8]; zdot = state[9]
        p = state[10];  q = state[11];  r = state[12]
        wM1 = state[13]; wdotM1 = state[14]
        wM2 = state[15]; wdotM2 = state[16]
        wM3 = state[17]; wdotM3 = state[18]
        wM4 = state[19]; wdotM4 = state[20]

        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*cmd[0]) / tau**2
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*cmd[1]) / tau**2
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*cmd[2]) / tau**2
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*cmd[3]) / tau**2

        wMotor = np.clip(np.array([wM1, wM2, wM3, wM4]), minWmotor, maxWmotor)
        thrust = kTh * wMotor**2
        torque = kTo * wMotor**2
        ThrM1, ThrM2, ThrM3, ThrM4 = thrust
        TorM1, TorM2, TorM3, TorM4 = torque

        velW, qW1, qW2 = wind.randomWind(t)

        # ─── Equations of motion (analytically solved from Newton-Euler) ───
        if config.orient == "NED":
            DynamicsDot = np.array([
                xdot, ydot, zdot,
                -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r,
                 0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r,
                 0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r,
                -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r,
                (Cd*sign(velW*cos(qW1)*cos(qW2)-xdot)*(velW*cos(qW1)*cos(qW2)-xdot)**2
                 - 2*(q0*q2+q1*q3)*(ThrM1+ThrM2+ThrM3+ThrM4)) / mB,
                (Cd*sign(velW*sin(qW1)*cos(qW2)-ydot)*(velW*sin(qW1)*cos(qW2)-ydot)**2
                 + 2*(q0*q1-q2*q3)*(ThrM1+ThrM2+ThrM3+ThrM4)) / mB,
                (-Cd*sign(velW*sin(qW2)+zdot)*(velW*sin(qW2)+zdot)**2
                 - (ThrM1+ThrM2+ThrM3+ThrM4)*(q0**2-q1**2-q2**2+q3**2) + g*mB) / mB,
                ((IByy-IBzz)*q*r - uP*IRzz*(wM1-wM2+wM3-wM4)*q + (ThrM1-ThrM2-ThrM3+ThrM4)*dym) / IBxx,
                ((IBzz-IBxx)*p*r + uP*IRzz*(wM1-wM2+wM3-wM4)*p + (ThrM1+ThrM2-ThrM3-ThrM4)*dxm) / IByy,
                ((IBxx-IByy)*p*q - TorM1+TorM2-TorM3+TorM4) / IBzz,
            ])
        else:  # ENU
            DynamicsDot = np.array([
                xdot, ydot, zdot,
                -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r,
                 0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r,
                 0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r,
                -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r,
                (Cd*sign(velW*cos(qW1)*cos(qW2)-xdot)*(velW*cos(qW1)*cos(qW2)-xdot)**2
                 + 2*(q0*q2+q1*q3)*(ThrM1+ThrM2+ThrM3+ThrM4)) / mB,
                (Cd*sign(velW*sin(qW1)*cos(qW2)-ydot)*(velW*sin(qW1)*cos(qW2)-ydot)**2
                 - 2*(q0*q1-q2*q3)*(ThrM1+ThrM2+ThrM3+ThrM4)) / mB,
                (-Cd*sign(velW*sin(qW2)+zdot)*(velW*sin(qW2)+zdot)**2
                 + (ThrM1+ThrM2+ThrM3+ThrM4)*(q0**2-q1**2-q2**2+q3**2) - g*mB) / mB,
                ((IByy-IBzz)*q*r + uP*IRzz*(wM1-wM2+wM3-wM4)*q + ( ThrM1-ThrM2-ThrM3+ThrM4)*dym) / IBxx,
                ((IBzz-IBxx)*p*r - uP*IRzz*(wM1-wM2+wM3-wM4)*p + (-ThrM1-ThrM2+ThrM3+ThrM4)*dxm) / IByy,
                ((IBxx-IBzz)*p*q + TorM1-TorM2+TorM3-TorM4) / IBzz,
            ])

        sdot       = np.zeros(21)
        sdot[0:13] = DynamicsDot
        sdot[13]   = wdotM1;  sdot[14] = wddotM1
        sdot[15]   = wdotM2;  sdot[16] = wddotM2
        sdot[17]   = wdotM3;  sdot[18] = wddotM3
        sdot[19]   = wdotM4;  sdot[20] = wddotM4

        self.acc = sdot[7:10]
        return sdot

    def update(self, t, Ts, cmd, wind):
        """Advance the simulation by one timestep Ts.

        Parameters
        ----------
        t : float
            Current time (s)
        Ts : float
            Timestep (s)
        cmd : ndarray (4,)
            Motor speed commands (rad/s)
        wind : Wind
            Wind disturbance model
        """
        prev_vel   = self.vel.copy()
        prev_omega = self.omega.copy()

        self.integrator.set_f_params(cmd, wind)
        self.state = self.integrator.integrate(t, t + Ts)

        self.pos    = self.state[0:3]
        self.quat   = self.state[3:7]
        self.vel    = self.state[7:10]
        self.omega  = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15],
                                self.state[17], self.state[19]])

        self.vel_dot   = (self.vel   - prev_vel)   / Ts
        self.omega_dot = (self.omega - prev_omega) / Ts

        self.extended_state()
        self.forces()
