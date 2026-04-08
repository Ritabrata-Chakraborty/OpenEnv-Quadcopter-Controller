# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quadnav Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import QuadnavAction, QuadnavObservation, QuadnavState


class QuadnavEnv(EnvClient[QuadnavAction, QuadnavObservation, QuadnavState]):
    """Client for the Quadnav Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated session (and physics state).

    Episode lifecycle
    -----------------
    1. Call ``reset(difficulty=...)`` to start an episode and receive the
       first observation.  Difficulty is 'easy', 'medium', or 'hard'; each
       selects a random map+goal from that tier.
    2. Call ``step(action)`` repeatedly.  Each ``StepResult`` contains:
       - ``observation`` — 40 LiDAR bins + 5 navigation scalars
       - ``reward``      — shaped scalar reward for this step
       - ``done``        — **True when the episode has ended**
    3. When ``done=True``, call ``state()`` to read ``outcome``:
       - ``'success'``  — drone reached the goal (dist < 1 m)
       - ``'crash'``    — collision, extreme tilt, or altitude fault
       - ``'timeout'``  — 600 steps / 60 s elapsed without reaching goal
    4. Call ``reset()`` again to begin the next episode.

    Example:
        >>> with QuadnavEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(task="easy")
        ...     while not result.done:
        ...         action = QuadnavAction(vx=0.5, vy=0.0, yaw_rate=0.0)
        ...         result = env.step(action)
        ...     state = env.state()
        ...     print("Episode ended:", state.outcome, "steps:", state.step_count)
        ...     # → Episode ended: success  steps: 47
    """

    def _step_payload(self, action: QuadnavAction) -> Dict:
        return {
            "vx": action.vx,
            "vy": action.vy,
            "yaw_rate": action.yaw_rate,
        }

    def _parse_result(self, payload: Dict) -> StepResult[QuadnavObservation]:
        obs_data = payload.get("observation", {})
        observation = QuadnavObservation(
            lidar_bins=obs_data.get("lidar_bins", []),
            goal_dist=obs_data.get("goal_dist", 0.0),
            goal_angle=obs_data.get("goal_angle", 0.0),
            prev_vx=obs_data.get("prev_vx", 0.0),
            prev_vy=obs_data.get("prev_vy", 0.0),
            prev_yaw_rate=obs_data.get("prev_yaw_rate", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> QuadnavState:
        return QuadnavState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "easy"),
            map_id=payload.get("map_id", 0),
            map_seed=payload.get("map_seed", 0),
            pos_x=payload.get("pos_x", 0.0),
            pos_y=payload.get("pos_y", 0.0),
            pos_z=payload.get("pos_z", 0.0),
            roll=payload.get("roll", 0.0),
            pitch=payload.get("pitch", 0.0),
            yaw=payload.get("yaw", 0.0),
            vel_x=payload.get("vel_x", 0.0),
            vel_y=payload.get("vel_y", 0.0),
            vel_z=payload.get("vel_z", 0.0),
            goal_x=payload.get("goal_x", 0.0),
            goal_y=payload.get("goal_y", 0.0),
            goal_dist=payload.get("goal_dist", 0.0),
            elapsed_time=payload.get("elapsed_time", 0.0),
            last_reward=payload.get("last_reward", 0.0),
            outcome=payload.get("outcome", "running"),
            score=payload.get("score", 0.0),
        )
