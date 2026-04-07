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

# Quadnav: Quadcopter Point-Navigation Environment

![Navigation Demo](assets/DDPG_Example.gif)

## Overview

**Quadnav** is a physics-based quadcopter navigation environment designed for training and evaluating autonomous navigation agents. 

The environment features a **realistic 6-DOF rigid-body simulator** with a **velocity-based controller** that accepts three normalized control inputs—forward, lateral, and angular velocity—to navigate from start positions to goal locations using:
- A 360° LiDAR scan (40 binned rays) for obstacle detection
- Normalized distance to goal
- Heading error relative to goal

Agents (RL policies, LLMs, classical planners, etc.) operate directly on these observations and command the velocity controller, which handles the low-level physics and motor dynamics.

---

## Simulation & Physics

The quadcopter dynamics are simulated by the `quadnav.sim.controller` package — a full 6-DOF rigid-body simulator grounded in Newton-Euler equations with second-order motor dynamics.

**Key features:**
- **State:** 21-element vector (position, quaternion, velocities, motor speeds)
- **Control:** 4-layer cascade (position → velocity → attitude → rate → motor mixer)
- **Trajectories:** Polynomial minimum jerk/snap trajectories
- **Wind:** Configurable disturbance models (None, Fixed, Sine, Random)  
- **Sensor:** 40-bin LiDAR-like raycasting at 12 m range

For detailed documentation on the physics engine, see [sim/README.md](sim/README.md).

---

## Action Space

**`QuadnavAction`** — one action per step, three continuous normalized values:

| Field | Type | Range | Physical Meaning |
|-------|------|-------|------------------|
| `vx` | float | [0, 1] | Forward velocity (0 = stop, 1 = ~3 m/s) |
| `vy` | float | [-1, 1] | Lateral velocity (left/right strafe) |
| `yaw_rate` | float | [-1, 1] | Angular velocity (~60°/s max) |

Altitude is held constant by the low-level flight controller; the agent only controls horizontal motion and heading.

---

## Observation Space

**`QuadnavObservation`** — returned after every `reset()` and `step()`:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `lidar_bins` | list[float] | 40 | Normalised LiDAR range [0, 1], one per angular bin |
| `goal_dist` | float | 1 | Distance to goal (normalised by sensor range) |
| `goal_angle` | float | 1 | Bearing to goal in robot frame (normalised [-1, 1]) |
| `prev_vx` | float | 1 | Previous step's vx command |
| `prev_vy` | float | 1 | Previous step's vy command |
| `prev_yaw_rate` | float | 1 | Previous step's yaw_rate command |

**Total observation vector: 45 elements** (40 LiDAR + 5 context)

---

## State

**`QuadnavState`** — full environment state, accessible via `env.state()`:

| Field | Type | Description |
|-------|------|-------------|
| `pos_x`, `pos_y`, `pos_z` | float | World position (metres) |
| `roll`, `pitch`, `yaw` | float | Euler angles (radians) |
| `vel_x`, `vel_y`, `vel_z` | float | Velocities (m/s) |
| `goal_x`, `goal_y` | float | Goal position (world frame) |
| `goal_dist` | float | Euclidean distance to goal (metres) |
| `elapsed_time` | float | Episode duration (seconds) |
| `outcome` | str | Episode result: `"success"`, `"crash"`, `"timeout"`, `"running"` |

---

## Reward Function

The reward is shaped to encourage **fast**, **safe**, and **goal-directed** navigation.

### Base Reward Equation

```
reward = r_distance + r_forward_vel + r_lateral_vel + r_angular_vel + r_yaw_align + r_obstacle + r_step_penalty + r_terminal
```

### Reward Components

| Component | Computation | Purpose |
|-----------|-----------|---------|
| **Distance** | `2 × d₀ / (d₀ + d) − 1` | Bounded progress: increases as agent approaches goal. d₀ = initial distance, d = current. |
| **Forward Vel** | `−(1 − vₓ / vₘₐₓ)²` | Encourage forward velocity close to max for active progress |
| **Lateral Vel** | `−vᵧ²` | Penalize inefficient lateral motion (strafe) |
| **Angular Vel** | `−ψ̇²` | Penalize excessive yaw rate (favor smooth turning) |
| **Yaw Align** | `−\|θ_goal\|` | Penalize heading misalignment with goal bearing |
| **Obstacle** | `−20` if `d_obs < 1.0m`, else `0` | Proximity penalty when obstacles detected nearby |
| **Step Cost** | `−1` per timestep | Incentivize completion speed |
| **Terminal** | `+2500 / −2000 / −100` | Success / Crash / Timeout |

**Example:** An agent that reaches the goal quickly in an obstacle-free area receives a strong positive cumulative reward.

---

## Tasks

Three difficulty levels are available. Each uses different maps and obstacle density:

| Task | Obstacle Density | Map Size | Time Limit | Typical Difficulty |
|------|-----------------|----------|-----------|-------------------|
| **Easy** | Low (open field) | 200×200 m | 60 s | Simple corridor navigation |
| **Medium** | Medium (cluttered) | 200×200 m | 40 s | Dense obstacles, narrow passages |
| **Hard** | High (maze-like) | <200×200 m | 60 s | Complex topology with dead-ends |

Each episode:
- Agent spawns at a fixed start location
- Goal is randomly selected within the map
- Agent must navigate using only LiDAR, distance, and angle observations
- Episode terminates on success, crash, or timeout

---


## Docker Deployment

### Build and Run Locally

```bash
# 1. Clean up old images and containers
docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null
docker rmi -f $(docker images -q) 2>/dev/null

# 2. Build the image (from project root, one level up)
cd ..
docker build -t quadnav:latest quadnav/
cd quadnav

# 3. Start the server (passes .env file for API credentials)
docker run -d \
  --name quadnav-env \
  -p 8000:8000 \
  --env-file .env \
  -e QUADNAV_ENV_URL=http://localhost:8000 \
  quadnav:latest

# 4. Wait for server to initialize
sleep 15

# 5. Run inference (e.g., LLM-driven navigation)
docker exec -w /app/env quadnav-env bash -c "uv run python3 inference.py"

# 6. Cleanup
docker stop quadnav-env && docker rm quadnav-env
```

### Deploy to Hugging Face Spaces

The Dockerfile is automatically built and deployed to HF Spaces:

```bash
git add .
git commit -m "Update Quadnav"
git push  # HF builds and deploys automatically
```

Access the web interface at: `https://huggingface.co/spaces/14372-Ritabrata/quadnav`

---

## Project Structure

```
quadnav/
├── __init__.py                       # Package entry point
├── models.py                         # Pydantic models (Action, Observation, State)
├── client.py                         # Client-side environment wrapper
├── inference.py                      # LLM-driven inference loop
├── openenv.yaml                      # OpenEnv task manifest
├── pyproject.toml                    # Dependencies and metadata
├── Dockerfile                        # Container build config
├── README.md                         # This file
│
├── server/
│   ├── app.py                        # FastAPI application
│   ├── environment.py                # Environment server (OpenEnv wrapper)
│   └── tasks.py                      # Async request handlers
│
├── sim/
│   ├── README.md                     # Physics engine documentation
│   ├── env.py                        # Gym-style environment interface
│   ├── parameter.py                  # Physics parameters
│   ├── dataset/
│   │   ├── easy/                     # Easy task maps and goals
│   │   ├── medium/                   # Medium task maps and goals
│   │   └── hard/                     # Hard task maps and goals
│   └── controller/                   # Quadcopter physics and control
│       ├── vehicle/                  # 6-DOF rigid-body simulator
│       ├── control.py                # 4-layer control cascade
│       └── utils/                    # Quaternions, rotations, wind models
│
├── assets/
│   └── point_goal_nav.gif            # Demonstration video
└── tests/                            # Unit and integration tests
```

---

## Quick Start

### 1. Start the environment server

```bash
docker run -d -p 8000:8000 --env-file .env quadnav:latest
```

The server is now available at `http://localhost:8000`.

### 2. Connect and run an episode

```python
from quadnav.client import QuadnavEnv
from quadnav.models import QuadnavAction

async with QuadnavEnv(base_url="http://localhost:8000") as env:
    obs = await env.reset(task="easy")
    
    for step in range(600):
        # Example: simple forward motion
        action = QuadnavAction(vx=0.7, vy=0.0, yaw_rate=0.0)
        obs = await env.step(action)
        
        print(f"Step {step}: Distance={obs.goal_dist:.2f}, Reward={obs.reward:.2f}")
        
        if obs.done:
            print(f"Episode Ended: {obs.done}")
            break
```

### 3. Web Interface (Optional)

Visit `http://localhost:8000` to visualize the environment in real-time.

---