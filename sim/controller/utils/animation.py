# -*- coding: utf-8 -*-
"""3D animation of a simulated quadcopter trajectory."""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

import quadnav.sim.controller.utils as utils
import quadnav.sim.controller.config as config

numFrames = 8  # render every N-th timestep


def sameAxisAnimation(
    t_all,
    waypoints,
    pos_all,
    quat_all,
    sDes_tr_all,
    Ts,
    params,
    xyzType,
    yawType,
    ifsave,
    start_pos=None,
    goal_pos=None,
    sphere_radius=None,
):
    """Animate the quadcopter trajectory on a shared 3-D axis.

    Parameters
    ----------
    t_all       : ndarray (T,)   — time vector (s)
    waypoints   : ndarray (N,3)  — mission waypoints (m)
    pos_all     : ndarray (T,3)  — position history (m)
    quat_all    : ndarray (T,4)  — quaternion history [w,x,y,z]
    sDes_tr_all : ndarray (T,19) — trajectory setpoint history
    Ts          : float           — physics timestep (s)
    params      : dict            — vehicle geometry (dxm, dym, dzm in m)
    xyzType     : int             — position trajectory selector (see run_simulation.py)
    yawType     : int             — yaw trajectory selector
    ifsave      : bool            — save animation as GIF if True
    start_pos   : array_like (3,) or None — draw a green sphere at this position
    goal_pos    : array_like (3,) or None — draw a red sphere at this position
    sphere_radius: float or None  — radius of start/goal spheres (m)

    Returns
    -------
    line_ani : matplotlib.animation.FuncAnimation
    """
    x = pos_all[:, 0]
    y = pos_all[:, 1]
    z = pos_all[:, 2]

    x_wp = waypoints[:, 0]
    y_wp = waypoints[:, 1]
    z_wp = waypoints[:, 2]

    if config.orient == "NED":
        z    = -z
        z_wp = -z_wp

    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # ─── Optional start/goal spheres (drawn first so quad lines render on top) ───
    if sphere_radius is not None and float(sphere_radius) > 0.0:
        u_s = np.linspace(0.0, 2.0 * np.pi, 20)
        v_s = np.linspace(0.0, np.pi, 12)
        uu, vv = np.meshgrid(u_s, v_s)
        r = float(sphere_radius)
        for center, color in ((start_pos, 'green'), (goal_pos, 'red')):
            if center is None:
                continue
            c = np.asarray(center, dtype=float).reshape(3)
            cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
            if config.orient == "NED":
                cz = -cz
            xs = cx + r * np.cos(uu) * np.sin(vv)
            ys = cy + r * np.sin(uu) * np.sin(vv)
            zs = cz + r * np.cos(vv)
            surf = ax.plot_surface(xs, ys, zs, color=color, alpha=0.45,
                                   linewidth=0, antialiased=True, shade=True)
            surf.set_zorder(1)

    line1, = ax.plot([], [], [], lw=2, color='red',  zorder=10)
    line2, = ax.plot([], [], [], lw=2, color='blue', zorder=10)
    line3, = ax.plot([], [], [], '--', lw=1, color='blue', zorder=9)

    # ─── Axis limits ───
    extraEachSide = 0.5
    maxRange = (0.5 * np.array([x.max()-x.min(),
                                y.max()-y.min(),
                                z.max()-z.min()]).max() + extraEachSide)
    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())

    ax.set_xlim3d([mid_x - maxRange, mid_x + maxRange]); ax.set_xlabel('X')
    if config.orient == "NED":
        ax.set_ylim3d([mid_y + maxRange, mid_y - maxRange])
    else:
        ax.set_ylim3d([mid_y - maxRange, mid_y + maxRange])
    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z - maxRange, mid_z + maxRange]); ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # ─── Trajectory type labels ───
    traj_labels = {
        0: 'Hover',
        1: 'Simple Waypoints',
        2: 'Simple Waypoint Interpolation',
        3: 'Minimum Velocity Trajectory',
        4: 'Minimum Acceleration Trajectory',
        5: 'Minimum Jerk Trajectory',
        6: 'Minimum Snap Trajectory',
        7: 'Minimum Acceleration Trajectory - Stop',
        8: 'Minimum Jerk Trajectory - Stop',
        9: 'Minimum Snap Trajectory - Stop',
        10: 'Minimum Jerk Trajectory - Fast Stop',
        11: 'Minimum Snap Trajectory - Fast Stop',
        12: 'Simple Waypoints',
    }
    yaw_labels = {0: 'None', 1: 'Waypoints', 2: 'Interpolation', 3: 'Follow', 4: 'Zero'}

    trajType    = traj_labels.get(xyzType, '')
    yawTrajType = yaw_labels.get(yawType, '')

    ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    ax.text2D(0.95, 0.91, 'Yaw: ' + yawTrajType, transform=ax.transAxes,
              horizontalalignment='right')

    def updateLines(i):
        time_i = t_all[i * numFrames]
        pos    = pos_all[i * numFrames]
        xi, yi, zi = pos[0], pos[1], pos[2]

        x_from0 = pos_all[0:i*numFrames, 0]
        y_from0 = pos_all[0:i*numFrames, 1]
        z_from0 = pos_all[0:i*numFrames, 2]

        dxm = params["dxm"]; dym = params["dym"]; dzm = params["dzm"]
        quat = quat_all[i * numFrames]

        if config.orient == "NED":
            zi      = -zi
            z_from0 = -z_from0
            quat    = np.array([quat[0], -quat[1], -quat[2], quat[3]])

        R = utils.quat2Dcm(quat)
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0],
                                 [dxm,  dym, dzm], [-dxm, dym, dzm],
                                 [0, 0, 0], [-dxm, -dym, dzm]])
        motorPoints = np.dot(R, motorPoints.T)
        motorPoints[0, :] += xi
        motorPoints[1, :] += yi
        motorPoints[2, :] += zi

        line1.set_data(motorPoints[0, 0:3], motorPoints[1, 0:3])
        line1.set_3d_properties(motorPoints[2, 0:3])
        line2.set_data(motorPoints[0, 3:6], motorPoints[1, 3:6])
        line2.set_3d_properties(motorPoints[2, 3:6])
        line3.set_data(x_from0, y_from0)
        line3.set_3d_properties(z_from0)
        titleTime.set_text(u"Time = {:.2f} s".format(time_i))
        return line1, line2, line3

    def ini_plot():
        for ln in (line1, line2, line3):
            ln.set_data(np.empty([1]), np.empty([1]))
            ln.set_3d_properties(np.empty([1]))
        return line1, line2, line3

    line_ani = animation.FuncAnimation(
        fig, updateLines, init_func=ini_plot,
        frames=len(t_all[0:-2:numFrames]),
        interval=(Ts * 1000 * numFrames),
        blit=False,
    )

    if ifsave:
        line_ani.save(
            'Gifs/Raw/animation_{0:.0f}_{1:.0f}.gif'.format(xyzType, yawType),
            dpi=80, writer='imagemagick', fps=25,
        )

    plt.show()
    return line_ani
