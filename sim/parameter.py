"""Environment parameters for quadcopter point-navigation."""

import math

# ------------------------------------------------------------------
# Sensor / state
# ------------------------------------------------------------------

NUM_SCAN_SAMPLES = 40
ACTION_SIZE = 3          # [linear_x, linear_y, angular_vel]

# ------------------------------------------------------------------
# Episode timing
# ------------------------------------------------------------------

PHYSICS_TS = 0.005                                      # physics sub-step (s)
DRL_STEP_DURATION = 0.1                                 # one agent step = 20 sub-steps (s)
MAX_EPISODE_STEP = 600                                  # steps per episode
EPISODE_TIMEOUT = MAX_EPISODE_STEP * DRL_STEP_DURATION  # 60 s

# ------------------------------------------------------------------
# Navigation thresholds
# ------------------------------------------------------------------

GOAL_THRESHOLD = 1.0       # metres — goal reached when dist < this
HOVER_ALTITUDE = -1.0      # metres (NED convention, negative = above ground)

# ------------------------------------------------------------------
# Map geometry
# ------------------------------------------------------------------

MAP_CELL_SIZE = 0.2                                          # metres per pixel
MAP_PIXELS = 250                                             # pixels per side
MAP_SIDE_M = (MAP_PIXELS - 1) * MAP_CELL_SIZE                # 62.25 m
MAX_GOAL_DISTANCE = math.hypot(MAP_SIDE_M, MAP_SIDE_M)       # diagonal upper bound
COLLISION_RADIUS = 0.40                                      # metres
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# ------------------------------------------------------------------
# Sensing
# ------------------------------------------------------------------

SENSOR_RANGE = 12.0        # metres — max LiDAR range

# ------------------------------------------------------------------
# Reward
# ------------------------------------------------------------------

REWARD_SUCCESS = 2500.0
REWARD_CRASH = -2000.0
REWARD_TIMEOUT = -100.0
OBSTACLE_PENALTY = -20.0
OBSTACLE_PENALTY_THRESHOLD = 0.80     # metres — penalty when closer than this

# ------------------------------------------------------------------
# Velocity limits
# ------------------------------------------------------------------

SPEED_LINEAR_MAX = 3.0                # m/s  forward
SPEED_LINEAR_Y_MAX = 1.5              # m/s  lateral
SPEED_ANGULAR_MAX = math.radians(60)  # rad/s

# ------------------------------------------------------------------
# Critic mode (server always uses single-critic)
# ------------------------------------------------------------------

USE_MULTI_CRITIC = False
