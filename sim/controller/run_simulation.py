# -*- coding: utf-8 -*-
"""Standalone 3D quadcopter simulation runner with visualization."""

import numpy as np
import matplotlib.pyplot as plt
import time

from quadnav.sim.controller.trajectory import Trajectory
from quadnav.sim.controller.control import Control
from quadnav.sim.controller.vehicle.quadcopter import Quadcopter
from quadnav.sim.controller.utils.wind import Wind
import quadnav.sim.controller.utils as utils
from quadnav.sim.controller.utils.animation import sameAxisAnimation
import quadnav.sim.controller.config as config


def quad_sim(t, Ts, quad, ctrl, wind, traj):
    """Advance the simulation by one timestep.

    Parameters
    ----------
    t : float
        Current time (s)
    Ts : float
        Timestep (s)
    quad : Quadcopter
        Vehicle state
    ctrl : Control
        Controller object
    wind : Wind
        Wind disturbance model
    traj : Trajectory
        Trajectory generator

    Returns
    -------
    t : float
        Updated time (s)
    """
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts
    sDes = traj.desiredState(t, Ts, quad)
    ctrl.controller(traj, quad, sDes, Ts)
    return t


def main():
    """Configure, run, and visualise a quadcopter simulation."""
    start_time = time.time()

    Ti     = 0
    Ts     = 0.005
    Tf     = 20
    ifsave = 0

    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect  = np.zeros(3)

    ctrlType = ctrlOptions[1]

    trajSelect[0] = 1
    trajSelect[1] = 0
    trajSelect[2] = 1

    print("Control type: {}".format(ctrlType))

    quad = Quadcopter(Ti)
    traj = Trajectory(quad, ctrlType, trajSelect)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    sDes = traj.desiredState(0, Ts, quad)
    ctrl.controller(traj, quad, sDes, Ts)

    numTimeStep = int(Tf / Ts + 1)

    t_all         = np.zeros(numTimeStep)
    s_all         = np.zeros([numTimeStep, len(quad.state)])
    pos_all       = np.zeros([numTimeStep, len(quad.pos)])
    vel_all       = np.zeros([numTimeStep, len(quad.vel)])
    quat_all      = np.zeros([numTimeStep, len(quad.quat)])
    omega_all     = np.zeros([numTimeStep, len(quad.omega)])
    euler_all     = np.zeros([numTimeStep, len(quad.euler)])
    sDes_traj_all = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all     = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all    = np.zeros([numTimeStep, len(quad.wMotor)])
    thr_all       = np.zeros([numTimeStep, len(quad.thr)])
    tor_all       = np.zeros([numTimeStep, len(quad.tor)])

    t_all[0]           = Ti
    s_all[0, :]        = quad.state
    pos_all[0, :]      = quad.pos
    vel_all[0, :]      = quad.vel
    quat_all[0, :]     = quad.quat
    omega_all[0, :]    = quad.omega
    euler_all[0, :]    = quad.euler
    sDes_traj_all[0,:] = traj.sDes
    sDes_calc_all[0,:] = ctrl.sDesCalc
    w_cmd_all[0, :]    = ctrl.w_cmd
    wMotor_all[0, :]   = quad.wMotor
    thr_all[0, :]      = quad.thr
    tor_all[0, :]      = quad.tor

    t = Ti
    i = 1
    while round(t, 3) < Tf:
        t = quad_sim(t, Ts, quad, ctrl, wind, traj)

        t_all[i]           = t
        s_all[i, :]        = quad.state
        pos_all[i, :]      = quad.pos
        vel_all[i, :]      = quad.vel
        quat_all[i, :]     = quad.quat
        omega_all[i, :]    = quad.omega
        euler_all[i, :]    = quad.euler
        sDes_traj_all[i,:] = traj.sDes
        sDes_calc_all[i,:] = ctrl.sDesCalc
        w_cmd_all[i, :]    = ctrl.w_cmd
        wMotor_all[i, :]   = quad.wMotor
        thr_all[i, :]      = quad.thr
        tor_all[i, :]      = quad.tor

        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all,
                      euler_all, w_cmd_all, wMotor_all, thr_all, tor_all,
                      sDes_traj_all, sDes_calc_all)
    ani = sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all,
                            Ts, quad.params, traj.xyzType, traj.yawType, ifsave)
    plt.show()


if __name__ == "__main__":
    if config.orient in ("NED", "ENU"):
        main()
    else:
        raise Exception("{} is not a valid orientation. Verify config.py file.".format(config.orient))
