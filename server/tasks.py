"""
tasks.py — Navigation task definitions and graders for the Quadnav environment.

Three tasks spanning easy → medium → hard difficulty:
  easy   — simple indoor maps,  short obstacles, relatively open navigation
  medium — denser indoor maps,  moderate obstacle density
  hard   — outdoor terrain maps, complex obstacle patterns, same 60 s budget

All tasks share the same episode budget: 600 steps × 0.1 s = 60 s.

Each reset() call randomly selects a map from the relevant difficulty tier
(dataset/maps_train/{easy,medium,hard}/), so graders must work on any map.

Grader signature:
    (outcome, initial_dist, final_dist, steps, max_steps) -> float in [0.0, 1.0]

    outcome:      'success' | 'crash' | 'timeout'
    initial_dist: normalised goal distance at episode start  (goal_dist from obs)
    final_dist:   normalised goal distance at episode end
    steps:        number of steps taken
    max_steps:    step budget
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

MAX_STEPS = 600

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

def _progress(initial_dist: float, final_dist: float) -> float:
    """Fraction of initial distance closed, clamped to [0, 1]."""
    if initial_dist <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - final_dist / initial_dist))


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _grade_easy(
    outcome: str,
    initial_dist: float,
    final_dist: float,
    steps: int,
    max_steps: int,
) -> float:
    """
    Easy grader — open indoor maps.

    success → 1.0
    crash   → partial credit (up to 0.3) based on distance progress
    timeout → partial credit (up to 0.8) based on distance progress
    """
    if outcome == "success":
        return 1.0
    prog = _progress(initial_dist, final_dist)
    if outcome == "crash":
        return round(prog * 0.3, 4)
    # timeout
    return round(prog * 0.8, 4)


def _grade_medium(
    outcome: str,
    initial_dist: float,
    final_dist: float,
    steps: int,
    max_steps: int,
) -> float:
    """
    Medium grader — denser indoor maps.

    success → efficiency-weighted score in [0.75, 1.0]
              (finishing faster gives a higher score)
    crash   → 0.0
    timeout → partial credit (up to 0.55) based on distance progress
    """
    if outcome == "success":
        efficiency = 1.0 - (steps / max_steps) * 0.25
        return round(min(1.0, max(0.75, efficiency)), 4)
    if outcome == "crash":
        return 0.0
    prog = _progress(initial_dist, final_dist)
    return round(prog * 0.55, 4)


def _grade_hard(
    outcome: str,
    initial_dist: float,
    final_dist: float,
    steps: int,
    max_steps: int,
) -> float:
    """
    Hard grader — outdoor terrain maps.

    success → efficiency-weighted score in [0.7, 1.0]
    crash   → 0.0
    timeout → partial credit (up to 0.3) based on distance progress
    """
    if outcome == "success":
        score = 1.0 - (steps / max_steps) * 0.3
        return round(min(1.0, max(0.7, score)), 4)
    if outcome == "crash":
        return 0.0
    prog = _progress(initial_dist, final_dist)
    return round(prog * 0.3, 4)


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Immutable descriptor for a single navigation task."""

    name: str
    description: str
    difficulty: Difficulty   # passed to reset(difficulty=...) on the server
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
        max_steps=MAX_STEPS,
        grader=_grade_easy,
    ),
    Task(
        name="medium",
        description=(
            "Navigate to the single goal on a randomly selected dense indoor map. "
            "Crashes receive no credit; faster success scores higher."
        ),
        difficulty="medium",
        max_steps=MAX_STEPS,
        grader=_grade_medium,
    ),
    Task(
        name="hard",
        description=(
            "Navigate to the single goal on a randomly selected outdoor terrain map. "
            "Only meaningful progress or successful completion scores above zero."
        ),
        difficulty="hard",
        max_steps=MAX_STEPS,
        grader=_grade_hard,
    ),
]

_TASK_MAP: dict[str, Task] = {t.name: t for t in TASKS}


def get_task(name: str) -> Task:
    """Return the Task with the given name, or raise KeyError."""
    try:
        return _TASK_MAP[name]
    except KeyError:
        valid = ", ".join(_TASK_MAP)
        raise KeyError(f"Unknown task {name!r}. Valid names: {valid}") from None
