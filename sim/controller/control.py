# -*- coding: utf-8 -*-
"""PID cascade controller: position → velocity → attitude → rate → mixer.

Control modes (traj.ctrlType):
    "xyz_pos"      — full position control (outer: pos → vel → thr → att → rate)
    "xy_vel_z_pos" — altitude hold + XY velocity control
    "xyz_vel"      — pure 3-axis velocity control

The sDes vector layout (length 19):
    [0:3]  desired position   (m)
    [3:6]  desired velocity   (m/s)
    [6:9]  desired acceleration (m/s²)
    [9:12] desired thrust     (N, in world frame)
    [12:15] desired Euler [φ,θ,ψ] (rad)
    [15:18] desired body rates [p,q,r] (rad/s)
    [18]   desired yaw rate  (rad/s)

Gain constants:
    pos_P_gain: Position proportional gains (dimensionless)
    vel_P_gain: Velocity proportional gains (dimensionless)
    vel_D_gain: Velocity derivative gains (dimensionless)
    vel_I_gain: Velocity integral gains (dimensionless)
    att_P_gain: Attitude proportional gains (dimensionless)
    rate_P_gain: Rate proportional gains (dimensionless)
    rate_D_gain: Rate derivative gains (dimensionless)
    velMax: Per-axis max (m/s)
    velMaxAll: Total speed max (m/s)
    tiltMax: Max tilt angle (rad)
    rateMax: Max body rates (rad/s)
"""

import numpy as np
from numpy import pi, sin, cos, sqrt
from numpy.linalg import norm
import quadnav.sim.controller.utils as utils
import quadnav.sim.controller.config as config

deg2rad = pi / 180.0

pos_P_gain = np.array([1.0, 1.0, 1.0])

vel_P_gain = np.array([5.0, 5.0, 4.0])
vel_D_gain = np.array([0.5, 0.5, 0.5])
vel_I_gain = np.array([5.0, 5.0, 5.0])

att_P_gain = np.array([8.0, 8.0, 1.5])

rate_P_gain = np.array([1.5, 1.5, 1.0])
rate_D_gain = np.array([0.04, 0.04, 0.1])

velMax              = np.array([5.0, 5.0, 5.0])
velMaxAll           = 5.0
saturateVel_separetely = False

tiltMax = 50.0 * deg2rad
rateMax = np.array([200.0, 200.0, 150.0]) * deg2rad


class Control:
    """Cascade PID controller for quadcopter flight.

    Parameters
    ----------
    quad : Quadcopter
        Provides params and current state
    yawType : int
        0 disables yaw control (sets ψ gain to 0)
    """

    def __init__(self, quad, yawType):
        self.sDesCalc  = np.zeros(16)
        self.w_cmd     = np.ones(4) * quad.params["w_hover"]
        self.thr_int   = np.zeros(3)
        if yawType == 0:
            att_P_gain[2] = 0
        self.setYawWeight()
        self.pos_sp    = np.zeros(3)
        self.vel_sp    = np.zeros(3)
        self.acc_sp    = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp    = np.zeros(3)
        self.pqr_sp    = np.zeros(3)
        self.yawFF     = np.zeros(3)

    def controller(self, traj, quad, sDes, Ts):
        """Run one control cycle and update self.w_cmd.

        Parameters
        ----------
        traj : Trajectory
            Provides sDes and ctrlType
        quad : Quadcopter
            Current vehicle state
        sDes : ndarray (19,)
            Desired state vector
        Ts : float
            Timestep (s)
        """
        self.pos_sp[:]    = traj.sDes[0:3]
        self.vel_sp[:]    = traj.sDes[3:6]
        self.acc_sp[:]    = traj.sDes[6:9]
        self.thrust_sp[:] = traj.sDes[9:12]
        self.eul_sp[:]    = traj.sDes[12:15]
        self.pqr_sp[:]    = traj.sDes[15:18]
        self.yawFF[:]     = traj.sDes[18]

        if traj.ctrlType == "xyz_vel":
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif traj.ctrlType == "xy_vel_z_pos":
            self.z_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif traj.ctrlType == "xyz_pos":
            self.z_pos_control(quad, Ts)
            self.xy_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)

        self.w_cmd = utils.mixerFM(quad, norm(self.thrust_sp), self.rateCtrl)

        self.sDesCalc[0:3]   = self.pos_sp
        self.sDesCalc[3:6]   = self.vel_sp
        self.sDesCalc[6:9]   = self.thrust_sp
        self.sDesCalc[9:13]  = self.qd
        self.sDesCalc[13:16] = self.rate_sp

    def z_pos_control(self, quad, Ts):
        """P controller: altitude error → vertical velocity setpoint."""
        self.vel_sp[2] += pos_P_gain[2] * (self.pos_sp[2] - quad.pos[2])

    def xy_pos_control(self, quad, Ts):
        """P controller: horizontal position error → horizontal velocity setpoint."""
        self.vel_sp[0:2] += pos_P_gain[0:2] * (self.pos_sp[0:2] - quad.pos[0:2])

    def saturateVel(self):
        """Clamp velocity setpoint to velMaxAll (total norm) or per-axis velMax."""
        if saturateVel_separetely:
            self.vel_sp = np.clip(self.vel_sp, -velMax, velMax)
        else:
            totalVel_sp = norm(self.vel_sp)
            if totalVel_sp > velMaxAll:
                self.vel_sp = self.vel_sp / totalVel_sp * velMaxAll

    def z_vel_control(self, quad, Ts):
        """PID controller: vertical velocity error → vertical thrust setpoint.

        Gravity feed-forward (m·g) keeps the vehicle hovering at zero error.
        Anti-windup prevents integrator wind-up at thrust saturation.
        """
        vel_z_error = self.vel_sp[2] - quad.vel[2]
        if config.orient == "NED":
            thrust_z_sp = (vel_P_gain[2]*vel_z_error
                           - vel_D_gain[2]*quad.vel_dot[2]
                           + quad.params["mB"]*(self.acc_sp[2] - quad.params["g"])
                           + self.thr_int[2])
            uMax = -quad.params["minThr"]   # negated/swapped for NED frame
            uMin = -quad.params["maxThr"]
        else:  # ENU
            thrust_z_sp = (vel_P_gain[2]*vel_z_error
                           - vel_D_gain[2]*quad.vel_dot[2]
                           + quad.params["mB"]*(self.acc_sp[2] + quad.params["g"])
                           + self.thr_int[2])
            uMax = quad.params["maxThr"]
            uMin = quad.params["minThr"]

        stop_int_D = ((thrust_z_sp >= uMax and vel_z_error >= 0.0)
                      or (thrust_z_sp <= uMin and vel_z_error <= 0.0))
        if not stop_int_D:
            self.thr_int[2] += vel_I_gain[2] * vel_z_error * Ts * quad.params["useIntergral"]
            self.thr_int[2]  = min(abs(self.thr_int[2]), quad.params["maxThr"]) * np.sign(self.thr_int[2])

        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    def xy_vel_control(self, quad, Ts):
        """PID controller: horizontal velocity error → horizontal thrust setpoint.

        Tracking anti-windup (Anti-Reset Windup) keeps integration bounded
        when the XY thrust vector is saturated by tilt or total thrust limits.
        """
        vel_xy_error  = self.vel_sp[0:2] - quad.vel[0:2]
        thrust_xy_sp  = (vel_P_gain[0:2]*vel_xy_error
                         - vel_D_gain[0:2]*quad.vel_dot[0:2]
                         + quad.params["mB"]*self.acc_sp[0:2]
                         + self.thr_int[0:2])

        thrust_max_xy_tilt = abs(self.thrust_sp[2]) * np.tan(tiltMax)
        thrust_max_xy      = sqrt(quad.params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy      = min(thrust_max_xy, thrust_max_xy_tilt)

        self.thrust_sp[0:2] = thrust_xy_sp
        if np.dot(self.thrust_sp[0:2], self.thrust_sp[0:2]) > thrust_max_xy**2:
            self.thrust_sp[0:2] = thrust_xy_sp / norm(thrust_xy_sp) * thrust_max_xy

        arw_gain    = 2.0 / vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2]) * arw_gain
        self.thr_int[0:2] += vel_I_gain[0:2] * vel_err_lim * Ts * quad.params["useIntergral"]

    def thrustToAttitude(self, quad, Ts):
        """Convert thrust vector + desired yaw into a full desired quaternion."""
        yaw_sp = self.eul_sp[2]

        body_z = -utils.vectNormalize(self.thrust_sp)
        if config.orient == "ENU":
            body_z = -body_z

        y_C    = np.array([-sin(yaw_sp), cos(yaw_sp), 0.0])
        body_x = utils.vectNormalize(np.cross(y_C, body_z))
        body_y = np.cross(body_z, body_x)
        R_sp   = np.array([body_x, body_y, body_z]).T

        self.qd_full = utils.RotToQuat(R_sp)

    def attitude_control(self, quad, Ts):
        """Quaternion-based attitude controller → body rate setpoint.

        Mixes a "reduced" quaternion (thrust-aligned, yaw-free) with the
        "full" quaternion (thrust + yaw) weighted by self.yaw_w, so that
        the tilt response is not degraded by the yaw channel.
        """
        e_z   = quad.dcm[:, 2]
        e_z_d = -utils.vectNormalize(self.thrust_sp)
        if config.orient == "ENU":
            e_z_d = -e_z_d

        qe_red      = np.zeros(4)
        qe_red[0]   = np.dot(e_z, e_z_d) + sqrt(norm(e_z)**2 * norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red      = utils.vectNormalize(qe_red)
        self.qd_red = utils.quatMultiply(qe_red, quad.quat)

        q_mix    = utils.quatMultiply(utils.inverse(self.qd_red), self.qd_full)
        q_mix    = q_mix * np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd  = utils.quatMultiply(
            self.qd_red,
            np.array([cos(self.yaw_w * np.arccos(q_mix[0])), 0, 0,
                      sin(self.yaw_w * np.arcsin(q_mix[3]))]),
        )

        self.qe      = utils.quatMultiply(utils.inverse(quad.quat), self.qd)
        self.rate_sp = (2.0 * np.sign(self.qe[0]) * self.qe[1:4]) * att_P_gain

        self.yawFF   = np.clip(self.yawFF, -rateMax[2], rateMax[2])
        self.rate_sp += utils.quat2Dcm(utils.inverse(quad.quat))[:, 2] * self.yawFF
        self.rate_sp  = np.clip(self.rate_sp, -rateMax, rateMax)

    def rate_control(self, quad, Ts):
        """PD controller: rate error → moment setpoint (self.rateCtrl)."""
        rate_error    = self.rate_sp - quad.omega
        self.rateCtrl = rate_P_gain * rate_error - rate_D_gain * quad.omega_dot

    def setYawWeight(self):
        """Compute yaw blending weight from attitude gains and normalise ψ gain."""
        roll_pitch_gain = 0.5 * (att_P_gain[0] + att_P_gain[1])
        self.yaw_w      = np.clip(att_P_gain[2] / roll_pitch_gain, 0.0, 1.0)
        att_P_gain[2]   = roll_pitch_gain
