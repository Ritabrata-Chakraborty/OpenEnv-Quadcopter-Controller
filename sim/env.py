"""Quadcopter point-navigation environment on 2D occupancy maps.

Wraps quadcopter physics with a Gym-like reset/step interface.
Provides raycasting, belief updates, and reward computation.
"""

import json
import os

import matplotlib
import numpy as np
from numpy import cos, pi, sin
from skimage import io as skio

mpl_backend = os.environ.get('MPLBACKEND', '').strip().lower()
if mpl_backend not in ('tkagg', 'qtagg', 'qt5agg', 'qt4agg', 'wxagg', 'wx'):
    matplotlib.use('Agg')

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

from quadnav.sim.parameter import (
    ACTION_SIZE,
    COLLISION_RADIUS,
    DRL_STEP_DURATION,
    EPISODE_TIMEOUT,
    FREE,
    GOAL_THRESHOLD,
    HOVER_ALTITUDE,
    MAP_CELL_SIZE,
    MAP_PIXELS,
    NUM_SCAN_SAMPLES,
    OBSTACLE_PENALTY,
    OBSTACLE_PENALTY_THRESHOLD,
    OCCUPIED,
    PHYSICS_TS,
    REWARD_CRASH,
    REWARD_SUCCESS,
    REWARD_TIMEOUT,
    SENSOR_RANGE,
    SPEED_ANGULAR_MAX,
    SPEED_LINEAR_MAX,
    SPEED_LINEAR_Y_MAX,
    UNKNOWN,
)
import quadnav.sim.utils as root_utils

compute_state = root_utils.compute_state
normalize_angle = root_utils.normalize_angle
bin_lidar = root_utils.bin_lidar

from quadnav.sim.controller.vehicle.quadcopter import Quadcopter
from quadnav.sim.controller.control import Control
from quadnav.sim.controller.utils.wind import Wind


# ------------------------------------------------------------------
# Coordinate helpers
# ------------------------------------------------------------------

def world_to_cell(world_x: float, world_y: float) -> tuple[int, int]:
    """Convert world coordinates (meters) to image pixel (col, row)."""
    col = int(round(world_x / MAP_CELL_SIZE + (MAP_PIXELS - 1) / 2.0))
    row = int(round((MAP_PIXELS - 1) / 2.0 - world_y / MAP_CELL_SIZE))
    return col, row


# ------------------------------------------------------------------
# Sensor model
# ------------------------------------------------------------------

def update_belief(
    robot_world_xy: np.ndarray,
    belief: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Reveal cells within SENSOR_RANGE around robot in the belief map."""
    col, row = world_to_cell(robot_world_xy[0], robot_world_xy[1])
    sr_cells = int(SENSOR_RANGE / MAP_CELL_SIZE)
    h, w = ground_truth.shape
    sr2 = sr_cells * sr_cells
    r_lo = max(0, row - sr_cells)
    r_hi = min(h, row + sr_cells + 1)
    c_lo = max(0, col - sr_cells)
    c_hi = min(w, col + sr_cells + 1)
    for r in range(r_lo, r_hi):
        dy = r - row
        for c in range(c_lo, c_hi):
            dx = c - col
            if dx * dx + dy * dy <= sr2:
                belief[r, c] = ground_truth[r, c]
    return belief


def raycast(
    robot_world_xy: np.ndarray,
    yaw: float,
    ground_truth: np.ndarray,
    num_rays: int = NUM_SCAN_SAMPLES,
    distances_only: bool = False,
) -> 'np.ndarray | tuple[np.ndarray, list, list]':
    """Cast evenly-spaced rays from robot position.

    When ``distances_only=True``, skips building endpoint/kind lists (saves RAM).
    """
    col, row = world_to_cell(robot_world_xy[0], robot_world_xy[1])
    sr_cells = int(SENSOR_RANGE / MAP_CELL_SIZE)
    h, w = ground_truth.shape
    distances = np.full(num_rays, SENSOR_RANGE, dtype=np.float32)

    if distances_only:
        for i in range(num_rays):
            angle = yaw + 2 * pi * i / num_rays
            dx = cos(angle)
            dy = -sin(angle)
            for step in range(1, sr_cells + 1):
                cx = int(round(col + dx * step))
                cy = int(round(row + dy * step))
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    distances[i] = step * MAP_CELL_SIZE
                    break
                if ground_truth[cy, cx] == OCCUPIED:
                    distances[i] = step * MAP_CELL_SIZE
                    break
        return distances

    endpoints = []
    hit_kinds = []

    for i in range(num_rays):
        angle = yaw + 2 * pi * i / num_rays
        dx = cos(angle)
        dy = -sin(angle)

        hit_col, hit_row = col, row
        ray_kind = 'max_range'
        for step in range(1, sr_cells + 1):
            cx = int(round(col + dx * step))
            cy = int(round(row + dy * step))
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                distances[i] = step * MAP_CELL_SIZE
                ray_kind = 'boundary'
                hit_col = int(np.clip(cx, 0, w - 1))
                hit_row = int(np.clip(cy, 0, h - 1))
                break
            if ground_truth[cy, cx] == OCCUPIED:
                distances[i] = step * MAP_CELL_SIZE
                ray_kind = 'obstacle'
                hit_col, hit_row = cx, cy
                break
            hit_col, hit_row = cx, cy

        endpoints.append((hit_col, hit_row))
        hit_kinds.append(ray_kind)

    return distances, endpoints, hit_kinds


# ------------------------------------------------------------------
# Velocity trajectory adapter
# ------------------------------------------------------------------

class CmdVelTrajectory:
    """Thin adapter that mimics the Trajectory interface for ctrl.controller().

    Uses 'xy_vel_z_pos' control mode: PID holds altitude, agent controls XY velocity.
    """

    def __init__(self, hover_altitude: float = HOVER_ALTITUDE):
        self.ctrlType = "xy_vel_z_pos"
        self.yawType = 1
        self.xyzType = 0
        self.sDes = np.zeros(19)
        self.sDes[2] = hover_altitude
        self.des_yaw = 0.0
        self.wps = np.array([[0.0, 0.0, hover_altitude]])

    def set_cmd_vel(self, linear_vel: float, lateral_vel: float, angular_vel: float, yaw: float, dt: float) -> None:
        """Set velocity command for one physics sub-step.

        linear_vel:  forward speed in body frame (>= 0).
        lateral_vel: lateral speed in body frame (positive = left).
        angular_vel: yaw rate (rad/s).
        """
        self.sDes[3] = linear_vel * cos(yaw) + lateral_vel * (-sin(yaw))
        self.sDes[4] = linear_vel * sin(yaw) + lateral_vel * cos(yaw)
        self.sDes[5] = 0.0
        self.des_yaw += angular_vel * dt
        self.des_yaw = normalize_angle(self.des_yaw)
        self.sDes[14] = self.des_yaw


# ------------------------------------------------------------------
# Physics / sensor helper functions (module-level globals)
# ------------------------------------------------------------------

def is_extreme_tilt(quad_euler: np.ndarray) -> bool:
    """True if roll or pitch exceeds 80 degrees."""
    return (abs(quad_euler[0]) > 80.0 * pi / 180.0
            or abs(quad_euler[1]) > 80.0 * pi / 180.0)


def check_collision(quad_pos: np.ndarray, ground_truth: np.ndarray) -> bool:
    """True if robot cell neighbourhood overlaps an obstacle or is out of bounds."""
    col, row = world_to_cell(quad_pos[0], quad_pos[1])
    h, w = ground_truth.shape
    if col < 0 or col >= w or row < 0 or row >= h:
        return True
    cr_cells = max(1, round(COLLISION_RADIUS / MAP_CELL_SIZE))
    for dy in range(-cr_cells, cr_cells + 1):
        for dx in range(-cr_cells, cr_cells + 1):
            ny, nx = row + dy, col + dx
            if 0 <= ny < h and 0 <= nx < w:
                if ground_truth[ny, nx] == OCCUPIED:
                    return True
    return False


def get_scan(
    quad_pos: np.ndarray,
    quad_euler: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Cast 360 rays, bin to NUM_SCAN_SAMPLES. Returns (bins_normalized, min_dist_m)."""
    distances = raycast(
        quad_pos[0:2], quad_euler[2], ground_truth,
        num_rays=360, distances_only=True,
    )
    bins = bin_lidar(distances, NUM_SCAN_SAMPLES, SENSOR_RANGE)
    return bins, float(np.min(distances))


def get_lidar(
    quad_pos: np.ndarray,
    quad_euler: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, list, list]:
    """Raycast for visualization (NUM_SCAN_SAMPLES rays)."""
    return raycast(quad_pos[0:2], quad_euler[2], ground_truth)


def compute_base_rewards(
    goal_dist: float,
    action: np.ndarray,
    min_obstacle_dist: float,
    initial_goal_dist: float,
    quad_pos: np.ndarray,
    quad_euler: np.ndarray,
    goal_pos: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Shared reward terms used by both single- and multi-critic modes.

    Returns (r_distance, r_vangular, r_vlinear, r_vlateral, r_obstacle, r_yaw).
    """
    r_distance = (2.0 * initial_goal_dist) / (initial_goal_dist + goal_dist) - 1.0
    r_vangular = -(action[2] ** 2)
    linear_vel = float(np.clip(action[0], 0.0, 1.0)) * SPEED_LINEAR_MAX
    r_vlinear = -((SPEED_LINEAR_MAX - linear_vel) / SPEED_LINEAR_MAX) ** 2
    r_vlateral = -(action[1] ** 2)
    r_obstacle = OBSTACLE_PENALTY if min_obstacle_dist < OBSTACLE_PENALTY_THRESHOLD else 0.0
    diff_xy = goal_pos - quad_pos[0:2]
    goal_angle = normalize_angle(np.arctan2(diff_xy[1], diff_xy[0]) - quad_euler[2])
    r_yaw = -abs(goal_angle)
    return r_distance, r_vangular, r_vlinear, r_vlateral, r_obstacle, r_yaw


def check_terminal(
    collision: bool,
    quad_euler: np.ndarray,
    quad_pos_z: float,
    total_time: float,
    goal_dist: float,
) -> tuple[float, bool, str]:
    """Evaluate terminal conditions.

    Returns:
        terminal_reward: additional reward for this transition.
        done: whether the episode ends.
        outcome: one of 'success', 'crash', 'timeout', or 'running'.
    """
    if collision or is_extreme_tilt(quad_euler) or quad_pos_z > 0.0:
        return REWARD_CRASH, True, 'crash'
    if goal_dist < GOAL_THRESHOLD:
        return REWARD_SUCCESS, True, 'success'
    if total_time >= EPISODE_TIMEOUT:
        return REWARD_TIMEOUT, True, 'timeout'
    return 0.0, False, 'running'


def compute_reward(
    goal_dist: float,
    action: np.ndarray,
    collision: bool,
    min_obstacle_dist: float,
    initial_goal_dist: float,
    quad_pos: np.ndarray,
    quad_euler: np.ndarray,
    goal_pos: np.ndarray,
    total_time: float,
) -> tuple[float, bool, str]:
    """Single-critic reward with obstacle proximity and terminal bonuses.

    Returns:
        reward: scalar reward for this step.
        done: whether the episode ends.
        outcome: 'success', 'crash', 'timeout', or 'running'.
    """
    r_distance, r_vangular, r_vlinear, r_vlateral, r_obstacle, r_yaw = compute_base_rewards(
        goal_dist, action, min_obstacle_dist, initial_goal_dist, quad_pos, quad_euler, goal_pos,
    )
    terminal, done, outcome = check_terminal(collision, quad_euler, quad_pos_z=quad_pos[2], total_time=total_time, goal_dist=goal_dist)
    reward = r_distance + r_obstacle + r_vangular + r_vlinear + r_vlateral + r_yaw - 1.0 + terminal
    return reward, done, outcome


# ------------------------------------------------------------------
# Quadcopter navigation environment
# ------------------------------------------------------------------

class QuadnavEnv:
    """Quadcopter point-navigation environment on 2D occupancy maps."""

    def __init__(self, maps_dir: str, goals_dir: str):
        self.maps_dir = maps_dir
        self.goals_dir = goals_dir
        self.Ts = PHYSICS_TS
        self.steps_per_action = int(DRL_STEP_DURATION / self.Ts)
        self.wind = Wind('None', 2.0, 90, -15)

        # Initialise all mutable state via reset()
        self.quad = None
        self.traj = None
        self.ctrl = None
        self.ground_truth = None
        self.robot_belief = None
        self.start_pos = None
        self.goal_pos = None
        self.map_name = None
        self.initial_goal_dist = 0.0
        self.t = 0.0
        self.travel_dist = 0.0
        self.prev_pos = None
        self.prev_action = np.zeros(ACTION_SIZE)
        self.total_time = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Map / goal loading
    # ------------------------------------------------------------------

    def load_map_and_goals(self) -> None:
        """Load a randomly selected map and its single goal."""
        map_list = sorted(
            f for f in os.listdir(self.maps_dir)
            if f.lower().endswith('.png')
        )
        if not map_list:
            raise FileNotFoundError(f"No PNG maps in {self.maps_dir}")
        map_file = map_list[np.random.randint(len(map_list))]
        self.map_name = os.path.splitext(map_file)[0]

        img = skio.imread(os.path.join(self.maps_dir, map_file))
        if img.ndim == 3:
            gray = np.mean(img[..., :3], axis=-1)
        else:
            gray = img.astype(float)
        self.ground_truth = np.where(gray > 200, FREE, OCCUPIED).astype(np.uint8)
        del img, gray

        json_path = os.path.join(self.goals_dir, self.map_name + '.json')
        with open(json_path) as f:
            data = json.load(f)
        self.start_pos = np.array(data['start_pose'], dtype=np.float64)
        # Each JSON has exactly one goal; always use index 0.
        self.goal_pos = np.array(data['goal_pose_list'][0], dtype=np.float64)

    def init_quad_at(self, x: float, y: float) -> None:
        """Place quadcopter at world (x, y) at hover altitude."""
        self.quad.state[0] = float(x)
        self.quad.state[1] = float(y)
        self.quad.state[2] = float(HOVER_ALTITUDE)
        self.quad.pos = self.quad.state[0:3].copy()
        self.quad.integrator.set_initial_value(
            np.asarray(self.quad.state, dtype=float), 0.0
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset to a new randomly selected map. Returns initial 45-dim state vector."""
        self.load_map_and_goals()
        self.robot_belief = np.full_like(self.ground_truth, UNKNOWN)

        self.quad = Quadcopter(0)
        self.init_quad_at(self.start_pos[0], self.start_pos[1])
        self.traj = CmdVelTrajectory()
        self.ctrl = Control(self.quad, self.traj.yawType)

        self.initial_goal_dist = float(np.linalg.norm(self.goal_pos - self.quad.pos[0:2]))

        self.t = 0.0
        self.travel_dist = 0.0
        self.prev_pos = self.quad.pos.copy()
        self.prev_action = np.zeros(ACTION_SIZE)
        self.total_time = 0.0

        update_belief(self.quad.pos[0:2], self.robot_belief, self.ground_truth)
        lidar_bins, _ = get_scan(self.quad.pos, self.quad.euler, self.ground_truth)
        state, _ = compute_state(lidar_bins, self.quad, self.goal_pos, self.prev_action)
        return state

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, str]:
        """Execute one DRL step (multiple physics sub-steps).

        Returns:
            state:   45-dim observation vector.
            reward:  scalar step reward.
            done:    True when the episode has ended (success / crash / timeout).
            outcome: 'success', 'crash', 'timeout', or 'running'.

        Call ``reset()`` immediately after receiving ``done=True``.
        """
        linear_vel = float(np.clip(action[0], 0.0, 1.0)) * SPEED_LINEAR_MAX  # forward [0, max]
        lateral_vel = action[1] * SPEED_LINEAR_Y_MAX                # bidirectional
        angular_vel = action[2] * SPEED_ANGULAR_MAX                 # bidirectional

        for _ in range(self.steps_per_action):
            yaw = self.quad.euler[2]
            self.traj.set_cmd_vel(linear_vel, lateral_vel, angular_vel, yaw, self.Ts)
            self.ctrl.controller(self.traj, self.quad, self.traj.sDes, self.Ts)
            self.quad.update(self.t, self.Ts, self.ctrl.w_cmd, self.wind)
            self.t += self.Ts

        self.travel_dist += float(np.linalg.norm(self.quad.pos - self.prev_pos))
        self.prev_pos = self.quad.pos.copy()
        self.total_time += DRL_STEP_DURATION

        update_belief(self.quad.pos[0:2], self.robot_belief, self.ground_truth)
        collision = check_collision(self.quad.pos, self.ground_truth)

        lidar_bins, min_obstacle_dist = get_scan(self.quad.pos, self.quad.euler, self.ground_truth)
        state, goal_dist = compute_state(lidar_bins, self.quad, self.goal_pos, action)
        reward, done, outcome = compute_reward(
            goal_dist, action, collision, min_obstacle_dist,
            self.initial_goal_dist, self.quad.pos, self.quad.euler,
            self.goal_pos, self.total_time,
        )

        self.prev_action = action.copy()
        return state, reward, done, outcome
