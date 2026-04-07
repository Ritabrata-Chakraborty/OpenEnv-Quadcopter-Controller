# -*- coding: utf-8 -*-
"""Trajectory generation for quadcopter waypoint following.

Supported position trajectory types (xyzType):
    0  — hover at origin
    1  — step to waypoint at each t_wps[i]
    2  — linear interpolation between waypoints
    3  — minimum velocity polynomial
    4  — minimum acceleration polynomial
    5  — minimum jerk polynomial
    6  — minimum snap polynomial
    7  — minimum acceleration, stop at each waypoint
    8  — minimum jerk, stop at each waypoint
    9  — minimum snap, stop at each waypoint
    10 — minimum jerk, momentary stop (fast stop)
    11 — minimum snap, momentary stop (fast stop)
    12 — advance to next waypoint on arrival
    13 — advance to next waypoint after dwell time
    99 — built-in XYZ test pattern

Supported yaw trajectory types (yawType):
    0 — no yaw control
    1 — step to y_wps[i] at each t_wps[i]
    2 — linearly interpolate between y_wps values
    3 — heading follows velocity direction
    4 — hold zero yaw

sDes vector layout (length 19):
    [0:3]  pos   (m), [3:6]  vel  (m/s), [6:9]  acc  (m/s²),
    [9:12] thr   (N), [12:15] eul (rad), [15:18] pqr (rad/s),
    [18]   yawRate (rad/s)
"""

import numpy as np
from numpy import pi
from numpy.linalg import norm
from quadnav.sim.controller.waypoints import makeWaypoints
import quadnav.sim.controller.config as config


class Trajectory:
    """Trajectory generator that produces sDes at each timestep.

    Parameters
    ----------
    quad : Quadcopter
        Used to read initial heading (psi)
    ctrlType : str
        One of "xyz_pos", "xy_vel_z_pos", "xyz_vel"
    trajSelect : array_like (3,)
        [xyzType, yawType, averVel]
        averVel: 0 = use t_wps timing, 1 = derive timing from average speed
    """

    def __init__(self, quad, ctrlType, trajSelect):
        self.ctrlType = ctrlType
        self.xyzType  = trajSelect[0]
        self.yawType  = trajSelect[1]
        self.averVel  = trajSelect[2]

        self.t_wps, self.wps, self.y_wps, self.v_wp = makeWaypoints()
        self.end_reached = 0

        if self.ctrlType == "xyz_pos":
            self.T_segment = np.diff(self.t_wps)

            if self.averVel == 1:
                dist = self.wps[1:] - self.wps[:-1]
                self.T_segment = np.sqrt((dist**2).sum(axis=1)) / self.v_wp
                self.t_wps     = np.zeros(len(self.T_segment) + 1)
                self.t_wps[1:] = np.cumsum(self.T_segment)

            if 3 <= self.xyzType <= 6:
                self.deriv_order = int(self.xyzType - 2)
                self.coeff_x = minSomethingTraj(self.wps[:, 0], self.T_segment, self.deriv_order)
                self.coeff_y = minSomethingTraj(self.wps[:, 1], self.T_segment, self.deriv_order)
                self.coeff_z = minSomethingTraj(self.wps[:, 2], self.T_segment, self.deriv_order)
            elif 7 <= self.xyzType <= 9:
                self.deriv_order = int(self.xyzType - 5)
                self.coeff_x = minSomethingTraj_stop(self.wps[:, 0], self.T_segment, self.deriv_order)
                self.coeff_y = minSomethingTraj_stop(self.wps[:, 1], self.T_segment, self.deriv_order)
                self.coeff_z = minSomethingTraj_stop(self.wps[:, 2], self.T_segment, self.deriv_order)
            elif 10 <= self.xyzType <= 11:
                self.deriv_order = int(self.xyzType - 7)
                self.coeff_x = minSomethingTraj_faststop(self.wps[:, 0], self.T_segment, self.deriv_order)
                self.coeff_y = minSomethingTraj_faststop(self.wps[:, 1], self.T_segment, self.deriv_order)
                self.coeff_z = minSomethingTraj_faststop(self.wps[:, 2], self.T_segment, self.deriv_order)

        if self.yawType == 4:
            self.y_wps = np.zeros(len(self.t_wps))

        self.current_heading = quad.psi

        self.desPos     = np.zeros(3)
        self.desVel     = np.zeros(3)
        self.desAcc     = np.zeros(3)
        self.desThr     = np.zeros(3)
        self.desEul     = np.zeros(3)
        self.desPQR     = np.zeros(3)
        self.desYawRate = 0.
        self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc,
                                self.desThr, self.desEul, self.desPQR,
                                self.desYawRate)).astype(float)

    def desiredState(self, t, Ts, quad):
        """Compute and return the 19-element sDes vector for simulation time t.

        Parameters
        ----------
        t : float
            Current time (s)
        Ts : float
            Timestep (s)
        quad : Quadcopter
            Current vehicle state
        """
        self.desPos     = np.zeros(3)
        self.desVel     = np.zeros(3)
        self.desAcc     = np.zeros(3)
        self.desThr     = np.zeros(3)
        self.desEul     = np.zeros(3)
        self.desPQR     = np.zeros(3)
        self.desYawRate = 0.

        def pos_waypoint_timed():
            if len(self.t_wps) != self.wps.shape[0]:
                raise Exception("Time array and waypoint array not the same size.")
            if (np.diff(self.t_wps) <= 0).any():
                raise Exception("Time array isn't properly ordered.")
            if t == 0:
                self.t_idx = 0
            elif t >= self.t_wps[-1]:
                self.t_idx = -1
            else:
                self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
            self.desPos = self.wps[self.t_idx, :]

        def pos_waypoint_interp():
            if len(self.t_wps) != self.wps.shape[0]:
                raise Exception("Time array and waypoint array not the same size.")
            if (np.diff(self.t_wps) <= 0).any():
                raise Exception("Time array isn't properly ordered.")
            if t == 0:
                self.t_idx = 0
                self.desPos = self.wps[0, :]
            elif t >= self.t_wps[-1]:
                self.t_idx = -1
                self.desPos = self.wps[-1, :]
            else:
                self.t_idx  = np.where(t <= self.t_wps)[0][0] - 1
                scale       = (t - self.t_wps[self.t_idx]) / self.T_segment[self.t_idx]
                self.desPos = (1 - scale)*self.wps[self.t_idx, :] + scale*self.wps[self.t_idx+1, :]

        def pos_waypoint_min():
            """Evaluate minimum-derivative polynomial at time t."""
            if len(self.t_wps) != self.wps.shape[0]:
                raise Exception("Time array and waypoint array not the same size.")
            nb_coeff = self.deriv_order * 2
            if t == 0:
                self.t_idx = 0
                self.desPos = self.wps[0, :]
            elif t >= self.t_wps[-1]:
                self.t_idx = -1
                self.desPos = self.wps[-1, :]
            else:
                self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
                scale = t - self.t_wps[self.t_idx]
                s = nb_coeff * self.t_idx
                e = nb_coeff * (self.t_idx + 1)
                t0 = get_poly_cc(nb_coeff, 0, scale)
                t1 = get_poly_cc(nb_coeff, 1, scale)
                t2 = get_poly_cc(nb_coeff, 2, scale)
                self.desPos = np.array([self.coeff_x[s:e].dot(t0),
                                        self.coeff_y[s:e].dot(t0),
                                        self.coeff_z[s:e].dot(t0)])
                self.desVel = np.array([self.coeff_x[s:e].dot(t1),
                                        self.coeff_y[s:e].dot(t1),
                                        self.coeff_z[s:e].dot(t1)])
                self.desAcc = np.array([self.coeff_x[s:e].dot(t2),
                                        self.coeff_y[s:e].dot(t2),
                                        self.coeff_z[s:e].dot(t2)])

        def pos_waypoint_arrived():
            dist_consider_arrived = 0.2  # m
            if t == 0:
                self.t_idx       = 0
                self.end_reached = 0
            elif not self.end_reached:
                d = norm(self.wps[self.t_idx] - quad.pos)
                if d < dist_consider_arrived:
                    self.t_idx += 1
                    if self.t_idx >= len(self.wps):
                        self.end_reached = 1
                        self.t_idx = -1
            self.desPos = self.wps[self.t_idx, :]

        def pos_waypoint_arrived_wait():
            dist_consider_arrived = 0.2  # m
            if t == 0:
                self.t_idx       = 0
                self.t_arrived   = 0
                self.arrived     = True
                self.end_reached = 0
            elif not self.end_reached:
                d = norm(self.wps[self.t_idx] - quad.pos)
                if d < dist_consider_arrived and not self.arrived:
                    self.t_arrived = t
                    self.arrived   = True
                elif self.arrived and (t - self.t_arrived > self.t_wps[self.t_idx]):
                    self.t_idx += 1
                    self.arrived = False
                    if self.t_idx >= len(self.wps):
                        self.end_reached = 0   # change to 1 to stop looping
                        self.t_idx = 0         # change to -1 to stop looping
            self.desPos = self.wps[self.t_idx, :]

        def yaw_waypoint_timed():
            if len(self.t_wps) != len(self.y_wps):
                raise Exception("Time array and yaw waypoint array not the same size.")
            self.desEul[2] = self.y_wps[self.t_idx]

        def yaw_waypoint_interp():
            if len(self.t_wps) != len(self.y_wps):
                raise Exception("Time array and yaw waypoint array not the same size.")
            if t == 0 or t >= self.t_wps[-1]:
                self.desEul[2] = self.y_wps[self.t_idx]
            else:
                scale          = (t - self.t_wps[self.t_idx]) / self.T_segment[self.t_idx]
                self.desEul[2] = (1-scale)*self.y_wps[self.t_idx] + scale*self.y_wps[self.t_idx+1]
                self.desYawRate        = (self.desEul[2] - self.current_heading) / Ts
                self.current_heading   = self.desEul[2]

        def yaw_follow():
            if self.xyzType in (1, 2, 12):
                if t == 0:
                    self.desEul[2] = 0
                else:
                    self.desEul[2] = np.arctan2(self.desPos[1]-quad.pos[1],
                                                 self.desPos[0]-quad.pos[0])
            elif self.xyzType == 13:
                if t == 0:
                    self.desEul[2] = 0
                    self.prevDesYaw = 0.
                else:
                    if not self.arrived:
                        self.desEul[2] = np.arctan2(self.desPos[1]-quad.pos[1],
                                                      self.desPos[0]-quad.pos[0])
                        self.prevDesYaw = self.desEul[2]
                    else:
                        self.desEul[2] = self.prevDesYaw
            else:
                if t == 0 or t >= self.t_wps[-1]:
                    self.desEul[2] = self.y_wps[self.t_idx]
                else:
                    self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])

            # Handle wrap-around from ±π without discontinuity in heading
            if (np.sign(self.desEul[2]) != np.sign(self.current_heading)
                    and abs(self.desEul[2] - self.current_heading) >= 2*pi - 0.1):
                self.current_heading += np.sign(self.desEul[2]) * 2*pi

            self.desYawRate      = (self.desEul[2] - self.current_heading) / Ts
            self.current_heading = self.desEul[2]

        if self.ctrlType in ("xyz_vel", "xy_vel_z_pos"):
            if self.xyzType == 1:
                self.sDes = testVelControl(t)

        elif self.ctrlType == "xyz_pos":
            if self.xyzType == 0:
                pass
            elif self.xyzType == 99:
                self.sDes = testXYZposition(t)
            else:
                if   self.xyzType == 1:                      pos_waypoint_timed()
                elif self.xyzType == 2:                      pos_waypoint_interp()
                elif 3 <= self.xyzType <= 11:                pos_waypoint_min()
                elif self.xyzType == 12:                     pos_waypoint_arrived()
                elif self.xyzType == 13:                     pos_waypoint_arrived_wait()

                if   self.yawType == 0:   pass
                elif self.yawType == 1:   yaw_waypoint_timed()
                elif self.yawType == 2:   yaw_waypoint_interp()
                elif self.yawType == 3:   yaw_follow()

                self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc,
                                       self.desThr, self.desEul, self.desPQR,
                                       self.desYawRate)).astype(float)

        return self.sDes


def get_poly_cc(n, k, t):
    """Return the coefficient vector for the k-th derivative of an n-th order polynomial at time t.

    Parameters
    ----------
    n : int
        Polynomial order
    k : int
        Derivative order (0 = position)
    t : float
        Time within segment (s)
    """
    assert n > 0 and k >= 0, "order and derivative must be positive."
    cc = np.ones(n)
    D  = np.linspace(n-1, 0, n)
    for i in range(n):
        for _ in range(k):
            cc[i] *= D[i]
            D[i]  -= 1
            if D[i] == -1:
                D[i] = 0
    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])
    return cc


def minSomethingTraj(waypoints, times, order):
    """Compute polynomial coefficients for a minimum-derivative trajectory.

    Boundary derivatives (up to order-1) are zero; intermediate derivatives
    are continuous. M = 2*order coefficients per segment.

    Parameters
    ----------
    waypoints : array_like (N+1,)
        Waypoint values for one axis
    times : array_like (N,)
        Segment durations (s)
    order : int
        Derivative order to minimise (1=vel, 2=acc, ...)

    Returns
    -------
    ndarray (M*N,)
        Polynomial coefficients
    """
    n        = len(waypoints) - 1
    nb_coeff = order * 2
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    for i in range(n):
        B[i]     = waypoints[i]
        B[i + n] = waypoints[i+1]

    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)]   = get_poly_cc(nb_coeff, 0, 0)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    for k in range(1, order):
        A[2*n+k-1][:nb_coeff]                  = get_poly_cc(nb_coeff, k, 0)
        A[2*n+(order-1)+k-1][-nb_coeff:]        = get_poly_cc(nb_coeff, k, times[-1])

    for i in range(n-1):
        for k in range(1, nb_coeff-1):
            A[2*n+2*(order-1)+i*2*(order-1)+k-1][i*nb_coeff:(i+1)*nb_coeff*2] = np.concatenate(
                (get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0))
            )

    return np.linalg.solve(A, B)


def minSomethingTraj_stop(waypoints, times, order):
    """Minimum-derivative trajectory with all derivatives zeroed at each waypoint.

    The drone comes to a momentary full stop (zero velocity, acceleration, …)
    at every waypoint before continuing. M = 2*order coefficients per segment.

    Parameters
    ----------
    waypoints : array_like (N+1,)
        Waypoint values
    times : array_like (N,)
        Segment durations (s)
    order : int
        Derivative order to minimise
    """
    n        = len(waypoints) - 1
    nb_coeff = order * 2
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    for i in range(n):
        B[i]     = waypoints[i]
        B[i + n] = waypoints[i+1]

    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)]   = get_poly_cc(nb_coeff, 0, 0)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    for i in range(n):
        for k in range(1, order):
            A[2*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)]             = get_poly_cc(nb_coeff, k, 0)
            A[2*n+(order-1)*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, k, times[i])

    return np.linalg.solve(A, B)


def minSomethingTraj_faststop(waypoints, times, order):
    """Minimum-derivative trajectory with zero velocity (only) at each waypoint.

    Velocity is forced to zero at each waypoint; higher derivatives remain
    continuous across waypoints so the drone leaves in the same direction it
    arrived. M = 2*order coefficients per segment.

    Parameters
    ----------
    waypoints : array_like (N+1,)
        Waypoint values
    times : array_like (N,)
        Segment durations (s)
    order : int
        Derivative order to minimise
    """
    n        = len(waypoints) - 1
    nb_coeff = order * 2
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    for i in range(n):
        B[i]     = waypoints[i]
        B[i + n] = waypoints[i+1]

    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)]   = get_poly_cc(nb_coeff, 0, 0)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])
        A[i+2*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, 0)
        A[i+3*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, times[i])

    for k in range(2, order):
        A[4*n+k-2][:nb_coeff]              = get_poly_cc(nb_coeff, k, 0)
        A[4*n+k-2+(order-2)][-nb_coeff:]   = get_poly_cc(nb_coeff, k, times[-1])

    for i in range(n-1):
        for k in range(2, nb_coeff-2):
            A[4*n+2*(order-2)+k-2+i*(nb_coeff-4)][i*nb_coeff:i*nb_coeff+nb_coeff*2] = np.concatenate(
                (get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0))
            )

    return np.linalg.solve(A, B)


def testXYZposition(t):
    """Simple XYZ position test: hover → move to [2,2,1] → move to [2,-2,-2] with yaw.
    
    Parameters
    ----------
    t : float
        Simulation time (s)
        
    Returns
    -------
    ndarray (19,)
        Desired state vector with units (m), (m/s), (m/s²), (N), (rad), (rad/s)
    """
    desPos = np.zeros(3); desVel = np.zeros(3); desAcc = np.zeros(3)
    desThr = np.zeros(3); desEul = np.zeros(3); desPQR = np.zeros(3)
    desYawRate = 30.0 * pi / 180.0

    if 1 <= t < 4:
        desPos = np.array([2., 2., 1.])
    elif t >= 4:
        desPos = np.array([2., -2., -2.])
        desEul = np.array([0., 0., pi/3])

    return np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)


def testVelControl(t):
    """Square velocity test: forward → sideways → 180° yaw → back → sideways.
    
    Parameters
    ----------
    t : float
        Simulation time (s)
        
    Returns
    -------
    ndarray (19,)
        Desired state vector with units (m), (m/s), (m/s²), (N), (rad), (rad/s)
    """
    desPos = np.zeros(3); desVel = np.zeros(3); desAcc = np.zeros(3)
    desThr = np.zeros(3); desEul = np.zeros(3); desPQR = np.zeros(3)
    desYawRate = 0.

    speed        = 2.0
    side_time    = 3.0
    yaw_turn     = pi
    yaw_duration = 2.0
    yaw_rate     = yaw_turn / yaw_duration

    desPos[2] = -1.0

    t1 = side_time
    t2 = t1 + side_time
    t3 = t2 + yaw_duration
    t4 = t3 + side_time
    t5 = t4 + side_time

    if t < t1:
        desVel[0] = speed
    elif t < t2:
        desVel[1] = speed
    elif t < t3:
        desEul[2]  = (t - t2) / yaw_duration * yaw_turn
        desYawRate = yaw_rate
    elif t < t4:
        desVel[0] = -speed
        desEul[2] = pi
    elif t < t5:
        desVel[1] = -speed
        desEul[2] = pi
    else:
        desEul[2] = pi

    return np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
