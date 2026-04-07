---
title: Quadnav Environment Server
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Quadnav Environment

A quadcopter point-navigation environment wrapped as an OpenEnv server. A simulated quadcopter must navigate 2D occupancy maps — floor plans, outdoor terrain grids, or procedurally generated obstacle fields — and reach a goal position as quickly and safely as possible. The environment is grounded in 3D physics via `quadnav.sim.controller` and uses a 40-bin LiDAR-like raycasting sensor that mimics the kind of rangefinder suites found on real autonomous vehicles, making learned policies directly transferable to physical systems.

This environment is useful for benchmarking local motion planning, obstacle avoidance, and goal-directed navigation under realistic kinematic constraints.

---

## Simulation & Physics

The quadcopter dynamics are simulated by the `quadnav.sim.controller` package — a full 6-DOF rigid-body simulator grounded in Newton-Euler equations with second-order motor dynamics.

For detailed documentation on the physics engine, control cascade, trajectory generation, state vectors, and wind models, see [sim/README.md](sim/README.md).

**Quick overview:**
- **State:** 21-element vector (position, quaternion, velocities, motor speeds)
- **Control:** 4-layer cascade (position → velocity → attitude → rate → motor mixer)
- **Trajectories:** Polynomial minima (velocity, accel, jerk, snap) with optional stops
- **Wind:** Configurable disturbance models (None, Fixed, Sine, Random)
- **Frames:** NED (North-East-Down) or ENU (East-North-Up) selectable

---

## Action Space

**`QuadnavAction`** — one action per step, three continuous values:

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `vx` | float | [0, 1] | Forward linear velocity (normalised, 0 = stop, 1 = full speed) |
| `vy` | float | [-1, 1] | Left/right lateral velocity (normalised) |
| `yaw_rate` | float | [-1, 1] | Clockwise/counter-clockwise rotation rate (normalised) |

Values are scaled internally to physical velocity commands before being applied to the simulator. Altitude is held constant by the low-level flight controller; the agent only controls horizontal motion.

---

## Observation Space

**`QuadnavObservation`** — returned after every `reset()` and `step()`:

| Field | Type | Description |
|-------|------|-------------|
| `lidar` | list[float] \[40\] | Normalised range readings, one per angular bin (0 = contact, 1 = max range) |
| `goal_dist` | float | Euclidean distance to goal in metres (normalised by sensor range) |
| `goal_angle` | float | Bearing to goal in the robot frame, radians in [-π, π] |
| `prev_vx` | float | Previous step's `vx` command |
| `prev_vy` | float | Previous step's `vy` command |
| `prev_yaw_rate` | float | Previous step's `yaw_rate` command |

Total observation vector length: **45** (40 LiDAR + 5 kinematic context).

---

## State

**`QuadnavState`** — full environment state, accessible via `env.state`:

| Field | Type | Description |
|-------|------|-------------|
| `pos_x`, `pos_y`, `pos_z` | float | World-frame position (metres) |
| `roll`, `pitch`, `yaw` | float | Euler angles (radians) |
| `vel_x`, `vel_y`, `vel_z` | float | Body-frame velocities (m/s) |
| `goal_x`, `goal_y` | float | Goal position in world frame |
| `goal_dist` | float | Current distance to goal (metres) |
| `elapsed_time` | float | Seconds elapsed in the current episode |
| `crashed` | bool | Whether the vehicle has collided with an obstacle |
| `success` | bool | Whether the goal has been reached |

---

## Reward Function

Reward is shaped to encourage fast and safe goal approach:

$$r_{\text{total}} = r_{\text{dist}} + r_{\text{vlin}} + r_{\text{vlat}} + r_{\text{vang}} + r_{\text{yaw}} + r_{\text{obs}} + r_{\text{live}} + r_{\text{term}}$$

| Term | Expression | Description |
|------|-----------|-------------|
| $r_{\text{dist}}$ | $\dfrac{2d_0}{d_0 + d} - 1$ | Bounded proximity reward; increases as agent approaches goal. $d_0$ = initial distance, $d$ = current distance. |
| $r_{\text{vlin}}$ | $-\left(\dfrac{v_{\max} - v_x}{v_{\max}}\right)^2$ | Encourages forward speed close to $v_{\max}$ (active progress toward goal). |
| $r_{\text{vlat}}$ | $-\dot{y}_t^2$ | Penalizes lateral motion $(v_y)$ for efficient, goal-directed flight. |
| $r_{\text{vang}}$ | $-\dot{\psi}_t^2$ | Penalizes excessive yaw rate $(\dot{\psi})$ for smooth heading control. |
| $r_{\text{yaw}}$ | $-\|\theta_{\text{goal}}\|$ | Penalizes heading misalignment with goal bearing. |
| $r_{\text{obs}}$ | $\begin{cases}\lambda_o & d_{\text{obs}} < \tau_o \\ 0 & \text{otherwise}\end{cases}$ | Obstacle proximity penalty when LiDAR distance $d_{\text{obs}}$ falls below threshold $\tau_o$. |
| $r_{\text{live}}$ | $-1$ | Per-step penalty (−1 per timestep) to incentivize faster episode completion. |
| $r_{\text{term}}$ | $\{+2500, -2000, -100\}$ | Terminal rewards: success, crash, timeout respectively. |

---

## Tasks

Three tasks of increasing difficulty are registered in `openenv.yaml`. Each uses a different occupancy map and starting configuration:

| Task | Environment | Density | Time Limit |
|------|-------------|---------|-----------|
| **Easy** | Open field (200×200 m), minimal obstacles | Low | 60 s |
| **Medium** | Cluttered corridor (200×200 m), moderate obstacles | Medium | 40 s |
| **Hard** | Complex map (< 200×200 m), dense obstacles | High | 60 s |

Each episode tasks the agent to navigate from a fixed start to a randomly selected goal, using only LiDAR observations and previous actions.

---

## Setup

### Build and Run with Docker

```bash
# From the quadnav_env directory
docker build -t quadnav-env:latest .

docker run -p 8000:8000 quadnav-env:latest
```

The server will be available at `http://localhost:8000`. The web UI is at `/web` and the API docs at `/docs`.


### Deploy to Hugging Face Spaces

```bash
# From the quadnav_env directory (where openenv.yaml lives)
openenv push
```

---

## Project Structure

```
quadnav/
├── __init__.py                       # Package root
├── models.py                         # Pydantic models: Action, Observation, State
├── client.py                         # QuadnavEnv (EnvClient subclass)
├── openenv.yaml                      # OpenEnv manifest and task definitions
├── pyproject.toml                    # Package metadata and dependencies
├── Dockerfile                        # Container image definition
├── README.md                         # This file
├── DOCKER_SETUP.md                   # Deployment & Docker guide
├── server/
│   ├── app.py                        # FastAPI application
│   ├── environment.py                # Environment wrapper
│   ├── tasks.py                      # Async task handlers
│   └── Dockerfile                    # Server build config
└── sim/
    ├── README.md                     # Physics engine documentation
    ├── env.py                        # Gym-style environment interface
    └── controller/                   # Quadcopter physics & control
        ├── config.py, trajectory.py, control.py, run_simulation.py, ...
        ├── vehicle/                  # Rigid-body dynamics integration
        └── utils/                    # Quaternions, rotations, wind, animation
```

For **physics engine details**, see [sim/README.md](sim/README.md).
