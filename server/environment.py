"""
Quadnav server-side environment.

Wraps the bundled QuadnavEnv (physics + LiDAR + reward) behind the
OpenEnv Environment[ActT, ObsT, StateT] interface.

Physics step loop
-----------------
Each call to step() sends a (vx, vy, yaw_rate) velocity command to the
physics engine, which runs ``steps_per_action`` sub-steps at
PHYSICS_TS (5 ms) for a total DRL step duration of 0.1 s.

Path resolution
---------------
All simulation code and map data live in ``quadnav/sim/`` which is
bundled with this package.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Sys-path setup — must happen BEFORE any other imports
# ---------------------------------------------------------------------------

# Find sim/ directory: either from PYTHONPATH or by navigating from __file__
_QUADRL_PATH = None

# Try PYTHONPATH first (set in Docker)
for path in sys.path:
    sim_candidate = os.path.join(path, "sim")
    if os.path.isdir(sim_candidate):
        _QUADRL_PATH = sim_candidate
        break

# Fallback: navigate from current file location
if not _QUADRL_PATH:
    _HERE = os.path.dirname(os.path.abspath(__file__))  # quadnav/server/
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))  # project root
    _QUADRL_PATH = os.path.join(_PROJECT_ROOT, "sim")

if _QUADRL_PATH not in sys.path:
    sys.path.insert(0, _QUADRL_PATH)

import numpy as np
import random

# ---------------------------------------------------------------------------
# Force single-critic mode BEFORE env.py is imported.
# env.py does `from parameter import USE_MULTI_CRITIC` at module load time,
# so we patch the parameter module first, then import env.
# ---------------------------------------------------------------------------

import parameter as _quad_param  # noqa: E402

_quad_param.USE_MULTI_CRITIC = False

# env.py does `from parameter import USE_MULTI_CRITIC` at load time, so the
# patch above is enough as long as env hasn't been cached yet.  The line below
# is a belt-and-suspenders fix for the already-cached case.
import env as _env_module  # noqa: E402

_env_module.USE_MULTI_CRITIC = False

from env import QuadnavEnv  # noqa: E402

# ---------------------------------------------------------------------------
# OpenEnv imports
# ---------------------------------------------------------------------------

from openenv.core.env_server.interfaces import Environment  # noqa: E402

try:
    from ..models import QuadnavAction, QuadnavObservation, QuadnavState
    from .tasks import get_task
except ImportError:
    from quadnav.models import QuadnavAction, QuadnavObservation, QuadnavState
    from quadnav.server.tasks import get_task

# Map/goal directories per difficulty tier.
_DATASET = os.path.join(_QUADRL_PATH, "dataset")
_DIRS: dict[str, tuple[str, str]] = {
    "easy":   (os.path.join(_DATASET, "easy"),   os.path.join(_DATASET, "easy_goals")),
    "medium": (os.path.join(_DATASET, "medium"), os.path.join(_DATASET, "medium_goals")),
    "hard":   (os.path.join(_DATASET, "hard"),   os.path.join(_DATASET, "hard_goals")),
}


class QuadnavEnvironment(Environment):
    """OpenEnv wrapper around the Quadnav point-navigation environment.

    Each WebSocket session gets its own QuadnavEnvironment instance (and
    therefore its own QuadnavEnv + physics state), so concurrent sessions are
    fully isolated.

    Difficulty is selected per-episode via reset(difficulty=...).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._step_count: int = 0
        self._last_reward: float = 0.0
        self._outcome: str = "running"
        self._done: bool = False

        # ─── Episode metadata (will be set by reset()) ───
        self._task: str = "easy"
        self._map_id: int = 0
        self._map_seed: int = 0

        # ─── Grading state (set by reset, used by _compute_score) ───
        self._initial_goal_dist: float = 0.0
        self._max_steps: int = 600

        maps_dir, goals_dir = _DIRS["easy"]
        self._env = QuadnavEnv(maps_dir=maps_dir, goals_dir=goals_dir)

        # ─── Initialize with proper reset to generate seeds & metadata ───
        # Using reset() instead of direct _env.reset() ensures seeds are
        # generated and episode metadata is properly populated.
        self.reset(task="easy")

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task: str = "easy",
        map_seed: int | None = None,
        **kwargs,
    ) -> QuadnavObservation:
        """Reset to a new episode with specified task and seeds.

        Args:
            task: Difficulty tier: 'easy', 'medium', or 'hard'.
            map_seed: Seed to select map within the task (0-indexed map_id).
                      If None, a random seed is generated.

        Returns:
            QuadnavObservation: First observation of the episode.
        """
        if task not in _DIRS:
            task = "easy"

        maps_dir, goals_dir = _DIRS[task]
        if maps_dir != self._env.maps_dir:
            self._env = QuadnavEnv(maps_dir=maps_dir, goals_dir=goals_dir)

        # ─── Generate seed if not provided ───
        map_seed = map_seed if map_seed is not None else random.randint(0, 2**31 - 1)

        # ─── Select map based on map_seed ───
        map_files = sorted([f for f in os.listdir(maps_dir) if f.endswith('.png')])
        map_id = map_seed % len(map_files) if map_files else 0

        # ─── Store episode metadata ───
        self._task = task
        self._map_id = map_id
        self._map_seed = map_seed

        self._step_count = 0
        self._last_reward = 0.0
        self._outcome = "running"
        self._done = False

        # ─── Set seed for reproducibility ───
        random.seed(map_seed)
        np.random.seed(map_seed)

        self._obs_vec = self._env.reset()

        # ─── Grading: store initial distance and step budget ───
        task_obj = get_task(task)
        self._max_steps = task_obj.max_steps
        self._initial_goal_dist = float(self._obs_vec[40])

        return self._build_obs(done=False)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: QuadnavAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> QuadnavObservation:
        """Execute one DRL step (covers ``steps_per_action`` physics sub-steps).

        The action (vx, vy, yaw_rate) is passed directly to QuadnavEnv.step()
        as a numpy array.  Reward is computed inside QuadnavEnv using the
        single-critic formula.

        Check ``observation.done`` (or ``StepResult.done``) after each call.
        When ``done=True``, ``state().outcome`` is 'success', 'crash', or
        'timeout' — call ``reset()`` to start a new episode.
        Calling step() when done raises RuntimeError.
        """
        if self._done:
            raise RuntimeError(
                f"Episode already ended with outcome='{self._outcome}'. "
                "Call reset() before stepping again."
            )
        action_array = np.array(
            [action.vx, action.vy, action.yaw_rate], dtype=np.float32
        )
        obs_vec, reward, done, outcome = self._env.step(action_array)

        self._obs_vec = obs_vec
        self._last_reward = float(reward)
        self._step_count += 1

        # Physics-level termination (crash, success, 60 s physics timeout)
        if done:
            self._outcome = outcome
            self._done = True
        # Server-side per-task step budget (e.g. medium = 400 steps)
        elif self._step_count >= self._max_steps:
            self._outcome = "timeout"
            self._done = True

        return self._build_obs(done=self._done)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def state(self) -> QuadnavState:
        """Full internal state: kinematics, episode metadata, and outcome."""
        quad = self._env.quad
        env = self._env
        goal_dist = float(np.linalg.norm(env.goal_pos - quad.pos[0:2]))

        return QuadnavState(
            episode_id=self._build_episode_id(),
            step_count=self._step_count,
            # ─── Episode metadata ───
            task=self._task,
            map_id=self._map_id,
            map_seed=self._map_seed,
            # ─── Position ───
            pos_x=float(quad.pos[0]),
            pos_y=float(quad.pos[1]),
            pos_z=float(quad.pos[2]),
            # ─── Orientation ───
            roll=float(quad.euler[0]),
            pitch=float(quad.euler[1]),
            yaw=float(quad.euler[2]),
            # ─── Velocity ───
            vel_x=float(quad.vel[0]),
            vel_y=float(quad.vel[1]),
            vel_z=float(quad.vel[2]),
            # ─── Goal ───
            goal_x=float(env.goal_pos[0]),
            goal_y=float(env.goal_pos[1]),
            goal_dist=goal_dist,
            # ─── Episode outcome ───
            elapsed_time=float(env.total_time),
            last_reward=self._last_reward,
            outcome=self._outcome,
            score=self._compute_score(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self) -> float:
        """Compute task-graded score using the server's grader.

        Returns 0.0 while the episode is still running.
        """
        if self._outcome == "running":
            return 0.0
        task_obj = get_task(self._task)
        final_goal_dist = float(self._obs_vec[40])
        return task_obj.grader(
            self._outcome,
            self._initial_goal_dist,
            final_goal_dist,
            self._step_count,
            task_obj.max_steps,
        )

    def _build_episode_id(self) -> str:
        """Construct a human-readable episode ID from metadata.

        Format: task_map(X)_seed(Z)
        Example: easy_map(3)_seed(1847291834)
        """
        return f"{self._task}_map({self._map_id})_seed({self._map_seed})"

    def _build_obs(self, done: bool) -> QuadnavObservation:
        """Construct a QuadnavObservation from the latest 45-element obs vector."""
        v = self._obs_vec
        return QuadnavObservation(
            lidar_bins=v[:40].tolist(),
            goal_dist=float(v[40]),
            goal_angle=float(v[41]),
            prev_vx=float(v[42]),
            prev_vy=float(v[43]),
            prev_yaw_rate=float(v[44]),
            done=done,
            reward=self._last_reward,
        )
