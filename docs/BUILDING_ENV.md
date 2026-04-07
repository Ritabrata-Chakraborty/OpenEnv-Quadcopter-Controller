# Building the Quadnav OpenEnv Environment

Step-by-step guide for wrapping `DRL-NAV/env.py` as an OpenEnv environment.

## 1. Study the Reference Implementation First

Before writing any code, read:
- [`OpenEnv/envs/echo_env/`](../../OpenEnv/envs/echo_env/) — simplest complete example
- [`OpenEnv/.claude/docs/PATTERNS.md`](../../OpenEnv/.claude/docs/PATTERNS.md) — canonical patterns
- [`openenv-course/module-4/README.md`](../../openenv-course/module-4/README.md) — building tutorial

## 2. Scaffold the Environment

```bash
# From root of OpenEnv_SST (or install openenv-core and use its CLI)
openenv init quad_nav_env
```

Or create the directory structure manually:

```
quad_nav_env/
├── __init__.py
├── models.py
├── client.py
├── openenv.yaml
├── pyproject.toml
└── server/
    ├── quad_nav_environment.py
    ├── app.py
    ├── requirements.txt
    └── Dockerfile
```

## 3. Define the Models (`models.py`)

Map DRL-NAV types to Pydantic:

```python
from pydantic import BaseModel, Field
import numpy as np
from typing import Optional

class QuadnavAction(BaseModel):
    """3-DOF velocity command: linear_x ∈ [-1,1], lateral_y ∈ [-1,1], angular ∈ [-1,1]."""
    class Config:
        arbitrary_types_allowed = True
    linear_x: float = Field(ge=-1.0, le=1.0)   # forward speed (scaled to SPEED_LINEAR_MAX)
    lateral_y: float = Field(ge=-1.0, le=1.0)   # lateral speed (scaled to SPEED_LINEAR_Y_MAX)
    angular: float = Field(ge=-1.0, le=1.0)      # yaw rate (scaled to SPEED_ANGULAR_MAX)

class QuadnavObservation(BaseModel):
    """State vector + episode metadata."""
    class Config:
        arbitrary_types_allowed = True
    state: list[float]          # length 45: [40 lidar bins, goal_dist, goal_angle, vx, vy, w]
    reward: float
    done: bool
    success: bool = False
    crash: bool = False
    timeout: bool = False
    step: int = 0

class QuadnavState(BaseModel):
    """Episode-level state for external monitoring."""
    episode_index: int
    map_name: str
    total_time: float
    travel_dist: float
    goal_dist: float = 0.0
```

## 4. Server Environment (`server/quad_nav_environment.py`)

The server imports `QuadnavEnv` from DRL-NAV and wraps it:

```python
import numpy as np
from openenv.core.env_server import Environment
from quad_nav_env.models import QuadnavAction, QuadnavObservation, QuadnavState

# DRL-NAV code lives next to this file in Docker; adjust import path accordingly
import sys, os
sys.path.insert(0, os.environ.get("DRL_NAV_PATH", "/app/DRL-NAV"))
from env import QuadnavEnv as _QuadnavEnv
import parameter as P

class QuadnavEnvironment(Environment[QuadnavAction, QuadnavObservation, QuadnavState]):
    def __init__(self):
        self._episode_index = 0
        self._env: _QuadnavEnv | None = None
        self._step_count = 0
        self._last_state = np.zeros(P.STATE_SIZE)

    def reset(self, seed=None, episode_id=None) -> QuadnavObservation:
        if seed is not None:
            np.random.seed(seed)
        self._env = _QuadnavEnv(self._episode_index)
        self._episode_index += 1
        self._step_count = 0
        raw_state = self._env.reset()
        self._last_state = raw_state
        return QuadnavObservation(state=raw_state.tolist(), reward=0.0, done=False)

    def step(self, action: QuadnavAction) -> QuadnavObservation:
        assert self._env is not None, "Call reset() before step()"
        act = np.array([action.linear_x, action.lateral_y, action.angular], dtype=np.float32)
        raw_state, reward, done, info = self._env.step(act)
        self._last_state = raw_state
        self._step_count += 1
        return QuadnavObservation(
            state=raw_state.tolist(),
            reward=float(reward),
            done=done,
            success=info.get("success", False),
            crash=info.get("crash", False),
            timeout=info.get("timeout", False),
            step=self._step_count,
        )

    @property
    def state(self) -> QuadnavState:
        env = self._env
        if env is None:
            return QuadnavState(episode_index=self._episode_index, map_name="", total_time=0.0, travel_dist=0.0)
        import numpy as np
        goal_dist = float(np.linalg.norm(env.goal_pos - env.quad.pos[0:2])) if env.goal_pos is not None else 0.0
        return QuadnavState(
            episode_index=self._episode_index,
            map_name=getattr(env, "map_name", ""),
            total_time=env.total_time,
            travel_dist=env.travel_dist,
            goal_dist=goal_dist,
        )
```

## 5. Dockerfile

Key: copy `DRL-NAV/` into the image alongside the env package.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for scipy, scikit-image, matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/lib/apt/lists/*

# Copy DRL-NAV research code
COPY DRL-NAV/ /app/DRL-NAV/

# Copy environment package
COPY quad_nav_env/ /app/quad_nav_env/
COPY quad_nav_env/server/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENV DRL_NAV_PATH=/app/DRL-NAV
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

EXPOSE 8000
CMD ["uvicorn", "quad_nav_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 6. Testing Locally

```bash
# Build
docker build -t quad-nav-env:latest -f quad_nav_env/server/Dockerfile .

# Run
docker run -p 8000:8000 quad-nav-env:latest

# Quick smoke test (Python)
python3 - <<'EOF'
from quad_nav_env import QuadnavEnv, QuadnavAction
with QuadnavEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    print("State size:", len(obs.observation.state))
    obs = env.step(QuadnavAction(linear_x=0.5, lateral_y=0.0, angular=0.1))
    print("Reward:", obs.reward, "Done:", obs.done)
EOF
```

## 7. Key Gotchas

- **`MPLBACKEND=Agg`** — `env.py` uses matplotlib; set non-interactive backend in Docker.
- **SimCon path collision** — `env.py` temporarily swaps `sys.modules['utils']` to avoid name clash; this is already handled in the original code, don't touch it.
- **`dataset/` maps** — must be present at runtime. Either `COPY` them into the image or mount them as a volume. The maps directory is ~50MB for the outdoor set.
- **Physics step count** — each `step()` call runs `steps_per_action = int(DRL_STEP_DURATION / PHYSICS_TS) = 20` physics sub-steps. This is CPU-bound; keep concurrency low for simulation envs.
- **No GPU needed** — default `USE_GPU=False`; inference/env stepping is CPU-only.
