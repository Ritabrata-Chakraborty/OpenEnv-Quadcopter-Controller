# -*- coding: utf-8 -*-
"""Plotting utilities for post-simulation result visualisation."""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import quadnav.sim.controller.utils as utils

rad2deg  = 180.0 / pi
rads2rpm = 60.0  / (2.0 * pi)


def fullprint(*args, **kwargs):
    """Print numpy arrays without truncation."""
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    print(*args, **kwargs)
    np.set_printoptions(opt)


def makeFigures(params, time, pos_all, vel_all, quat_all, omega_all, euler_all,
                commands, wMotor_all, thrust, torque, sDes_traj, sDes_calc):
    """Generate a standard set of diagnostic plots from a simulation run.

    Parameters
    ----------
    params     : dict        — vehicle parameters (unused here, kept for API compat)
    time       : ndarray (T,) — simulation time vector (s)
    pos_all    : ndarray (T,3) — position history (m)
    vel_all    : ndarray (T,3) — velocity history (m/s)
    quat_all   : ndarray (T,4) — quaternion history [w,x,y,z]
    omega_all  : ndarray (T,3) — body angular rate history (rad/s)
    euler_all  : ndarray (T,3) — Euler angle history [φ,θ,ψ] (rad)
    commands   : ndarray (T,4) — motor speed commands (rad/s)
    wMotor_all : ndarray (T,4) — actual motor speeds (rad/s)
    thrust     : ndarray (T,4) — per-motor thrust (N)
    torque     : ndarray (T,4) — per-motor torque (N·m)
    sDes_traj  : ndarray (T,19) — trajectory setpoint vector
    sDes_calc  : ndarray (T,16) — controller computed setpoint vector
    """
    # ─── Unpack position / velocity ───
    x, y, z       = pos_all[:,0], pos_all[:,1], pos_all[:,2]
    xdot, ydot, zdot = vel_all[:,0], vel_all[:,1], vel_all[:,2]

    # ─── Euler angles (deg) ───
    phi   = euler_all[:,0] * rad2deg
    theta = euler_all[:,1] * rad2deg
    psi   = euler_all[:,2] * rad2deg

    # ─── Body rates (deg/s) ───
    p = omega_all[:,0] * rad2deg
    q = omega_all[:,1] * rad2deg
    r = omega_all[:,2] * rad2deg

    # ─── Motor speeds (RPM) ───
    wM1 = wMotor_all[:,0] * rads2rpm
    wM2 = wMotor_all[:,1] * rads2rpm
    wM3 = wMotor_all[:,2] * rads2rpm
    wM4 = wMotor_all[:,3] * rads2rpm
    uM1 = commands[:,0]   * rads2rpm
    uM2 = commands[:,1]   * rads2rpm
    uM3 = commands[:,2]   * rads2rpm
    uM4 = commands[:,3]   * rads2rpm

    # ─── Controller setpoints ───
    x_sp, y_sp, z_sp       = sDes_calc[:,0], sDes_calc[:,1], sDes_calc[:,2]
    Vx_sp, Vy_sp, Vz_sp    = sDes_calc[:,3], sDes_calc[:,4], sDes_calc[:,5]
    x_thr_sp = sDes_calc[:,6]
    y_thr_sp = sDes_calc[:,7]
    z_thr_sp = sDes_calc[:,8]
    pDes = sDes_calc[:,13] * rad2deg
    qDes = sDes_calc[:,14] * rad2deg
    rDes = sDes_calc[:,15] * rad2deg

    # ─── Trajectory setpoints ───
    x_tr  = sDes_traj[:,0]; y_tr  = sDes_traj[:,1]; z_tr  = sDes_traj[:,2]
    Vx_tr = sDes_traj[:,3]; Vy_tr = sDes_traj[:,4]; Vz_tr = sDes_traj[:,5]
    Ax_tr = sDes_traj[:,6]; Ay_tr = sDes_traj[:,7]; Az_tr = sDes_traj[:,8]
    yaw_tr = sDes_traj[:,14] * rad2deg

    # ─── Desired Euler angles from desired quaternion ───
    psiDes   = np.zeros(sDes_calc.shape[0])
    thetaDes = np.zeros(sDes_calc.shape[0])
    phiDes   = np.zeros(sDes_calc.shape[0])
    for ii in range(sDes_calc.shape[0]):
        YPR = utils.quatToYPR_ZYX(sDes_calc[ii, 9:13])
        psiDes[ii]   = YPR[0] * rad2deg
        thetaDes[ii] = YPR[1] * rad2deg
        phiDes[ii]   = YPR[2] * rad2deg

    x_err = x_sp - x
    y_err = y_sp - y
    z_err = z_sp - z

    plt.show()

    # ─── Position ───
    plt.figure()
    plt.plot(time, x, time, y, time, z)
    plt.plot(time, x_sp, '--', time, y_sp, '--', time, z_sp, '--')
    plt.grid(True)
    plt.legend(['x', 'y', 'z', 'x_sp', 'y_sp', 'z_sp'])
    plt.xlabel('Time (s)'); plt.ylabel('Position (m)')
    plt.draw()

    # ─── Velocity ───
    plt.figure()
    plt.plot(time, xdot, time, ydot, time, zdot)
    plt.plot(time, Vx_sp, '--', time, Vy_sp, '--', time, Vz_sp, '--')
    plt.grid(True)
    plt.legend(['Vx', 'Vy', 'Vz', 'Vx_sp', 'Vy_sp', 'Vz_sp'])
    plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)')
    plt.draw()

    # ─── Desired thrust ───
    plt.figure()
    plt.plot(time, x_thr_sp, time, y_thr_sp, time, z_thr_sp)
    plt.grid(True)
    plt.legend(['x_thr_sp', 'y_thr_sp', 'z_thr_sp'])
    plt.xlabel('Time (s)'); plt.ylabel('Desired Thrust (N)')
    plt.draw()

    # ─── Euler angles ───
    plt.figure()
    plt.plot(time, phi, time, theta, time, psi)
    plt.plot(time, phiDes, '--', time, thetaDes, '--', time, psiDes, '--')
    plt.plot(time, yaw_tr, '-.')
    plt.grid(True)
    plt.legend(['roll', 'pitch', 'yaw', 'roll_sp', 'pitch_sp', 'yaw_sp', 'yaw_tr'])
    plt.xlabel('Time (s)'); plt.ylabel('Euler Angle (°)')
    plt.draw()

    # ─── Body rates ───
    plt.figure()
    plt.plot(time, p, time, q, time, r)
    plt.plot(time, pDes, '--', time, qDes, '--', time, rDes, '--')
    plt.grid(True)
    plt.legend(['p', 'q', 'r', 'p_sp', 'q_sp', 'r_sp'])
    plt.xlabel('Time (s)'); plt.ylabel('Angular Velocity (°/s)')
    plt.draw()

    # ─── Motor speeds ───
    plt.figure()
    plt.plot(time, wM1, time, wM2, time, wM3, time, wM4)
    plt.plot(time, uM1, '--', time, uM2, '--', time, uM3, '--', time, uM4, '--')
    plt.grid(True)
    plt.legend(['w1', 'w2', 'w3', 'w4'])
    plt.xlabel('Time (s)'); plt.ylabel('Motor Speed (RPM)')
    plt.draw()

    # ─── Rotor thrust and torque ───
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, thrust[:,0], time, thrust[:,1], time, thrust[:,2], time, thrust[:,3])
    plt.grid(True)
    plt.legend(['thr1', 'thr2', 'thr3', 'thr4'], loc='upper right')
    plt.xlabel('Time (s)'); plt.ylabel('Rotor Thrust (N)')
    plt.draw()
    plt.subplot(2, 1, 2)
    plt.plot(time, torque[:,0], time, torque[:,1], time, torque[:,2], time, torque[:,3])
    plt.grid(True)
    plt.legend(['tor1', 'tor2', 'tor3', 'tor4'], loc='upper right')
    plt.xlabel('Time (s)'); plt.ylabel('Rotor Torque (N·m)')
    plt.draw()

    # ─── Trajectory setpoints ───
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('Trajectory Setpoints')
    plt.plot(time, x_tr, time, y_tr, time, z_tr)
    plt.grid(True); plt.legend(['x', 'y', 'z'], loc='upper right')
    plt.xlabel('Time (s)'); plt.ylabel('Position (m)')
    plt.subplot(3, 1, 2)
    plt.plot(time, Vx_tr, time, Vy_tr, time, Vz_tr)
    plt.grid(True); plt.legend(['Vx', 'Vy', 'Vz'], loc='upper right')
    plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)')
    plt.subplot(3, 1, 3)
    plt.plot(time, Ax_tr, time, Ay_tr, time, Az_tr)
    plt.grid(True); plt.legend(['Ax', 'Ay', 'Az'], loc='upper right')
    plt.xlabel('Time (s)'); plt.ylabel('Acceleration (m/s²)')
    plt.draw()

    # ─── Position error ───
    plt.figure()
    plt.plot(time, x_err, time, y_err, time, z_err)
    plt.grid(True)
    plt.legend(['x error', 'y error', 'z error'])
    plt.xlabel('Time (s)'); plt.ylabel('Position Error (m)')
    plt.draw()
