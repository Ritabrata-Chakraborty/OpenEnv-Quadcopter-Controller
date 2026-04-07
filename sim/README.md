# quadnav.sim — Quadcopter Physics & Controller

A self-contained Python package that simulates a 6-DOF quadcopter with a full PID cascade controller, polynomial trajectory generator, wind disturbance models, and 3-D visualisation.

---

## Directory Structure

```
sim/
├── env.py                        # Gym-style environment wrapping the physics
└── controller/                   # Core physics + controller package
    ├── __init__.py
    ├── config.py                 # Global simulation flags (orient, usePrecession)
    ├── waypoints.py              # Default 6-waypoint mission definition
    ├── trajectory.py             # Trajectory generator (Trajectory class)
    ├── control.py                # PID cascade controller (Control class)
    ├── run_simulation.py         # Standalone runner with visualisation
    ├── vehicle/
    │   ├── __init__.py
    │   ├── params.py             # Physical parameters, hover equilibrium, initial state
    │   └── quadcopter.py         # Quadcopter dynamics ODE (Quadcopter class)
    └── utils/
        ├── __init__.py           # Re-exports all utility symbols
        ├── quaternion.py         # Quaternion algebra (multiply, inverse, normalise)
        ├── rotations.py          # DCM ↔ quaternion ↔ Euler conversions
        ├── state_conversions.py  # Body ↔ world velocity frame transforms
        ├── mixer.py              # Force-moment mixer (motor speed commands)
        ├── wind.py               # Wind disturbance models (Wind class)
        ├── display.py            # Time-history plotting (makeFigures)
        └── animation.py          # Real-time 3-D animation (sameAxisAnimation)
```

---

## Key Classes

### `Quadcopter` — `vehicle/quadcopter.py`

Full 6-DOF rigid-body simulation with second-order motor dynamics.

```python
quad = Quadcopter(Ti)          # Ti: initial time (s)
quad.update(t, Ts, w_cmd, wind)
```

Internally integrates the Newton-Euler equations of motion using the Dormand-Prince (dopri5) solver from `scipy.integrate.ode`. The 21-element state vector is described below.

### `Control` — `control.py`

Cascade PID controller: position → velocity → attitude → rate → motor mixer.

```python
ctrl = Control(quad, yawType)
ctrl.controller(traj, quad, sDes, Ts)
# result: ctrl.w_cmd  — motor speed commands (rad/s), shape (4,)
```

`yawType=0` disables the yaw channel (sets ψ gain to zero).

### `Trajectory` — `trajectory.py`

Generates a time-parameterised desired-state vector for the controller.

```python
traj = Trajectory(quad, ctrlType, trajSelect)
sDes = traj.desiredState(t, Ts, quad)   # returns ndarray (19,)
```

### `Wind` — `utils/wind.py`

Wind disturbance injected into the equations of motion.

```python
wind = Wind(windType, velW, qW1, qW2)
```

---

## Configuration — `config.py`

| Variable | Values | Effect |
|---|---|---|
| `orient` | `"NED"` / `"ENU"` | Reference frame. NED: z positive down; ENU: z positive up. |
| `usePrecession` | `True` / `False` | Enable gyroscopic precession torque from spinning rotors. |

---

## State Vector — 21 elements

| Indices | Symbol | Unit | Description |
|---|---|---|---|
| 0–2 | x, y, z | m | Position |
| 3–6 | q0, q1, q2, q3 | — | Quaternion [w, x, y, z] |
| 7–9 | ẋ, ẏ, ż | m/s | Linear velocity |
| 10–12 | p, q, r | rad/s | Body angular rates |
| 13, 14 | wM1, ẇM1 | rad/s, rad/s² | Motor 1 speed + derivative |
| 15, 16 | wM2, ẇM2 | rad/s, rad/s² | Motor 2 speed + derivative |
| 17, 18 | wM3, ẇM3 | rad/s, rad/s² | Motor 3 speed + derivative |
| 19, 20 | wM4, ẇM4 | rad/s, rad/s² | Motor 4 speed + derivative |

Motor numbering: M1 front-left, M2 front-right, M3 rear-right, M4 rear-left (clockwise from M1).

---

## Desired-State Vector (sDes) — 19 elements

| Indices | Description | Unit |
|---|---|---|
| 0–2 | Desired position | m |
| 3–5 | Desired velocity | m/s |
| 6–8 | Desired acceleration | m/s² |
| 9–11 | Desired thrust (world frame) | N |
| 12–14 | Desired Euler angles [φ, θ, ψ] | rad |
| 15–17 | Desired body rates [p, q, r] | rad/s |
| 18 | Desired yaw rate | rad/s |

---

## Control Cascade

```
Desired position
      │  (P)
      ▼
Desired velocity  <── saturateVel ───┐
      │  (PID + gravity FF)          │
      ▼                              │
Desired thrust vector                │
      │  thrustToAttitude            │
      ▼                              │
Desired quaternion                   │
      │  (P, reduced+full blend)     │
      ▼                              │
Desired body rates                   │
      │  (PD)                        │
      ▼
rateCtrl  ─>  mixerFM  ─>  w_cmd (rad/s per motor)
```

Control modes (set via `ctrlType`):

| Mode | Description |
|---|---|
| `"xyz_pos"` | Full position control |
| `"xy_vel_z_pos"` | Altitude hold + XY velocity commands |
| `"xyz_vel"` | Pure 3-axis velocity control |

---

## Position Trajectory Types (`trajSelect[0]`)

| Code | Name |
|---|---|
| 0 | Hover at start |
| 1 | Waypoint timed (explicit segment duration) |
| 2 | Waypoint interpolated (constant speed) |
| 3 | Minimum velocity polynomial |
| 4 | Minimum acceleration polynomial |
| 5 | Minimum jerk polynomial |
| 6 | Minimum snap polynomial |
| 7 | Minimum acceleration — stop at waypoints |
| 8 | Minimum jerk — stop at waypoints |
| 9 | Minimum snap — stop at waypoints |
| 10 | Minimum jerk — full stop (fast decel) |
| 11 | Minimum snap — full stop (fast decel) |
| 12 | Waypoint arrived (threshold-based switching) |
| 13 | Waypoint arrived + wait |

---

## Yaw Trajectory Types (`trajSelect[1]`)

| Code | Name |
|---|---|
| 0 | None (yaw locked) |
| 1 | Yaw waypoint timed |
| 2 | Yaw waypoint interpolated |
| 3 | Follow (yaw tracks velocity direction) |
| 4 | Zero (yaw driven to 0) |

---

## Waypoint Timing Mode (`trajSelect[2]`)

| Code | Behaviour |
|---|---|
| 0 | Use explicit waypoint times from `waypoints.py` |
| 1 | Derive segment time from `v_average` in `waypoints.py` |

---

## Wind Models

Constructed as `Wind(windType, velW, qW1, qW2)`.

| `windType` | Description |
|---|---|
| `'None'` | No wind |
| `'Fixed'` | Constant wind: speed `velW` (m/s), heading `qW1` (°), elevation `qW2` (°) |
| `'Sine'` | Sinusoidally varying wind (speed, heading, elevation each as sum of two sine waves) |
| `'Random'` | Like Sine but with randomly chosen amplitudes, frequencies, and phases |

`qW1` is the horizontal heading angle (° from x-axis); `qW2` is the elevation angle (°, positive up).

---

## Vehicle Parameters (defaults)

| Parameter | Value | Description |
|---|---|---|
| `mB` | 1.2 kg | Total mass |
| `g` | 9.81 m/s² | Gravitational acceleration |
| `dxm` / `dym` | 0.16 m | Arm length (x / y) |
| `IB` | diag(0.0123, 0.0123, 0.0224) kg·m² | Body inertia tensor |
| `kTh` | 1.076e-5 N/(rad/s)² | Thrust coefficient |
| `kTo` | 1.632e-7 N·m/(rad/s)² | Torque coefficient |
| `tau` | 0.015 s | Motor second-order time constant |
| `minThr` / `maxThr` | 0.4 / 36.72 N | Total thrust limits |
| `minWmotor` / `maxWmotor` | 75 / 925 rad/s | Motor speed limits |

---

## Integration with `env.py`

`sim/env.py` wraps the physics into a Gym-compatible interface for reinforcement-learning training:

```python
from quadnav.sim.controller.vehicle.quadcopter import Quadcopter
from quadnav.sim.controller.control import Control
from quadnav.sim.controller.utils.wind import Wind
```

The environment steps the `Quadcopter` ODE, runs the `Control` cascade, and returns observations and rewards suitable for policy training.
