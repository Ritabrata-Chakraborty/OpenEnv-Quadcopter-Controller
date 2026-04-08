"""
tasks.py — Navigation task definitions and graders for the Quadnav environment.

Three tasks spanning easy → medium → hard difficulty:
  easy   — simple indoor maps,  short obstacles, relatively open navigation (600 steps, 60 s)
  medium — denser indoor maps,  moderate obstacle density (400 steps, 40 s)
  hard   — outdoor terrain maps, complex obstacle patterns (600 steps, 60 s)

Each reset() call randomly selects a map from the relevant difficulty tier
(dataset/{easy,medium,hard}/), so graders must work on any map.

Grader signature:
    (outcome, initial_dist, final_dist, steps, max_steps) -> float in (0.0, 1.0)

    outcome:      'success' | 'crash' | 'timeout'
    initial_dist: normalised goal distance at episode start  (goal_dist from obs)
    final_dist:   normalised goal distance at episode end
    steps:        number of steps taken
    max_steps:    step budget
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

MAX_STEPS_DEFAULT = 600
MAX_STEPS_MEDIUM = 400

# Type alias
GraderFn = Callable[
    [
        str,    # outcome   — 'success' | 'crash' | 'timeout'
        float,  # initial_dist
        float,  # final_dist
        int,    # steps
        int,    # max_steps
    ],
    float,
]

Difficulty = Literal["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def clamp_score(score: float) -> float:
    """Clamp score strictly inside (0, 1) — validator rejects 0.0 and 1.0."""
    return round(min(0.9999, max(0.0001, score)), 4)


def distance_progress(initial_dist: float, final_dist: float) -> float:
    """Fraction of initial distance closed, clamped to [0, 1]."""
    if initial_dist <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - final_dist / initial_dist))


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_easy(
    outcome: str,
    initial_dist: float,
    final_dist: float,
    steps: int,
    max_steps: int,
) -> float:
    """Easy grader — open indoor maps.

    success → ~0.9999
    crash   → partial credit (up to 0.3) based on distance progress
    timeout → partial credit (up to 0.8) based on distance progress
    """
    if outcome == "success":
        return clamp_score(1.0)
    prog = distance_progress(initial_dist, final_dist)
    if outcome == "crash":
        return clamp_score(prog * 0.3)
    return clamp_score(prog * 0.8)


def grade_medium(
    outcome: str,
    initial_dist: float,
    final_dist: float,
    steps: int,
    max_steps: int,
) -> float:
    """Medium grader — denser indoor maps.

    success → efficiency-weighted score in [0.75, ~0.9999] (faster = higher)
    crash   → ~0.0001
    timeout → partial credit (up to 0.55) based on distance progress
    """
    if outcome == "success":
        efficiency = 1.0 - (steps / max_steps) * 0.25
        return clamp_score(min(1.0, max(0.75, efficiency)))
    if outcome == "crash":
        return clamp_score(0.0)
    return clamp_score(distance_progress(initial_dist, final_dist) * 0.55)


def grade_hard(
    outcome: str,
    initial_dist: float,
    final_dist: float,
    steps: int,
    max_steps: int,
) -> float:
    """Hard grader — outdoor terrain maps.

    success → efficiency-weighted score in [0.7, ~0.9999]
    crash   → ~0.0001
    timeout → partial credit (up to 0.3) based on distance progress
    """
    if outcome == "success":
        return clamp_score(min(1.0, max(0.7, 1.0 - (steps / max_steps) * 0.3)))
    if outcome == "crash":
        return clamp_score(0.0)
    return clamp_score(distance_progress(initial_dist, final_dist) * 0.3)


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Immutable descriptor for a single navigation task."""

    name: str
    description: str
    difficulty: Difficulty
    max_steps: int
    grader: GraderFn


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: list[Task] = [
    Task(
        name="easy",
        description=(
            "Navigate to the single goal on a randomly selected open indoor map. "
            "Partial credit is awarded for progress even on timeout."
        ),
        difficulty="easy",
        max_steps=MAX_STEPS_DEFAULT,
        grader=grade_easy,
    ),
    Task(
        name="medium",
        description=(
            "Navigate to the single goal on a randomly selected dense indoor map. "
            "Crashes receive no credit; faster success scores higher."
        ),
        difficulty="medium",
        max_steps=MAX_STEPS_MEDIUM,
        grader=grade_medium,
    ),
    Task(
        name="hard",
        description=(
            "Navigate to the single goal on a randomly selected outdoor terrain map. "
            "Only meaningful progress or successful completion scores above zero."
        ),
        difficulty="hard",
        max_steps=MAX_STEPS_DEFAULT,
        grader=grade_hard,
    ),
]

TASK_MAP: dict[str, Task] = {t.name: t for t in TASKS}


def get_task(name: str) -> Task:
    """Return the Task with the given name, or raise KeyError."""
    try:
        return TASK_MAP[name]
    except KeyError:
        valid = ", ".join(TASK_MAP)
        raise KeyError(f"Unknown task {name!r}. Valid names: {valid}") from None
