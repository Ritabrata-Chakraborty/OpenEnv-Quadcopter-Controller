# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Quadnav Environment.

Action: velocity command (vx, vy, yaw_rate), each normalized to [-1, 1].
Observation: 40 LiDAR bins + 5 navigation scalars (45 values total).
State: full internal drone state (position, rotation, velocity, episode info).
"""

from typing import List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class QuadnavAction(Action):
    """Velocity command for the quadcopter.

    Values are scaled to physical units inside the environment:
      vx       -> forward speed in [0, SPEED_LINEAR_MAX]      (maps from [0, 1])
      vy       -> lateral speed in [-SPEED_LINEAR_Y_MAX, SPEED_LINEAR_Y_MAX]
      yaw_rate -> angular rate in [-SPEED_ANGULAR_MAX, SPEED_ANGULAR_MAX]
    """

    vx: float = Field(..., description="Forward velocity, normalized [0, 1]")
    vy: float = Field(..., description="Lateral velocity, normalized [-1, 1]")
    yaw_rate: float = Field(..., description="Yaw rate, normalized [-1, 1]")


class QuadnavObservation(Observation):
    """Agent-visible observation: 40 LiDAR bins plus 5 navigation state values.

    The 45-element vector mirrors the STATE_SIZE used during RL training so
    that a trained policy can be applied directly to the server output.
    """

    lidar_bins: List[float] = Field(
        default_factory=list,
        description="40 normalized LiDAR distances, each in [0, 1]",
    )
    goal_dist: float = Field(
        default=0.0,
        description="Normalized distance to goal, in [0, 1]",
    )
    goal_angle: float = Field(
        default=0.0,
        description="Normalized angle to goal relative to drone heading, in [-1, 1]",
    )
    prev_vx: float = Field(default=0.0, description="Previous vx action")
    prev_vy: float = Field(default=0.0, description="Previous vy action")
    prev_yaw_rate: float = Field(default=0.0, description="Previous yaw_rate action")


class QuadnavState(State):
    """Full internal state of the quadcopter environment.

    Extends the base State (episode_id, step_count) with drone kinematics,
    episode metadata, and outcome flags.
    """

    # --- Episode metadata (for reproducibility & analysis) ---
    task: str = Field(
        default="easy",
        description="Difficulty tier: 'easy', 'medium', or 'hard'",
    )
    map_id: int = Field(
        default=0,
        description="Map index within the difficulty tier (0-indexed)",
    )
    map_seed: int = Field(
        default=0,
        description="Seed used to select the map",
    )

    # --- Position (world frame, meters) ---
    pos_x: float = Field(default=0.0, description="X position (m)")
    pos_y: float = Field(default=0.0, description="Y position (m)")
    pos_z: float = Field(default=0.0, description="Z position (m)")

    # --- Orientation (radians): roll, pitch, yaw ---
    roll: float = Field(default=0.0, description="Roll angle (rad)")
    pitch: float = Field(default=0.0, description="Pitch angle (rad)")
    yaw: float = Field(default=0.0, description="Yaw angle (rad)")

    # --- Velocity (world frame, m/s) ---
    vel_x: float = Field(default=0.0, description="X velocity (m/s)")
    vel_y: float = Field(default=0.0, description="Y velocity (m/s)")
    vel_z: float = Field(default=0.0, description="Z velocity (m/s)")

    # --- Goal info ---
    goal_x: float = Field(default=0.0, description="Goal X position (m)")
    goal_y: float = Field(default=0.0, description="Goal Y position (m)")
    goal_dist: float = Field(default=0.0, description="Euclidean distance to goal (m)")

    # --- Episode outcome ---
    elapsed_time: float = Field(default=0.0, description="Elapsed episode time (s)")
    last_reward: float = Field(default=0.0, description="Reward from the last step")
    outcome: str = Field(
        default="running",
        description="Episode outcome: 'running', 'success', 'crash', or 'timeout'",
    )
