"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import asyncio
import json
import os
import textwrap
from typing import Optional

from openai import OpenAI

from quadnav.client import QuadnavEnv
from quadnav.models import QuadnavAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL: str = os.environ.get("QUADNAV_ENV_URL", "http://localhost:8000")

TEMPERATURE: float = 0.0
MAX_TOKENS: int = 80

# LiDAR bin index below which we count a step as "near an obstacle"
# 0.15 * SENSOR_RANGE (12 m) ≈ 1.8 m
NEAR_OBSTACLE_THRESHOLD: float = 0.15

# ---------------------------------------------------------------------------
# Task definitions (easy → medium → hard)
# ---------------------------------------------------------------------------

TASKS = [
    dict(name="easy",   difficulty="easy",   max_steps=600),
    dict(name="medium", difficulty="medium", max_steps=400),
    dict(name="hard",   difficulty="hard",   max_steps=600),
]

# ---------------------------------------------------------------------------
# Grader
#
# Unified scoring formula:
#   score = 0.8 × progress + 0.1 × efficiency + 0.1 × safety
#
# Where:
#   progress   (0.80) – fraction of initial distance closed
#   efficiency (0.10) – step economy: 1 – steps_taken / max_steps
#   safety     (0.10) – fraction of steps spent away from obstacles
#
# Final score clamped to [0, 1] and rounded to 4 decimals.
# ---------------------------------------------------------------------------

def grade(
    task_name: str,
    outcome: str,
    initial_dist: float,
    final_dist: float,
    steps: int,
    max_steps: int,
    near_obstacle_steps: int,
) -> float:
    if initial_dist <= 0:
        return 0.0

    progress   = max(0.0, min(1.0, 1.0 - final_dist / initial_dist))
    efficiency = 1.0 - steps / max_steps   if steps > 0 else 1.0
    safety     = 1.0 - near_obstacle_steps / max(steps, 1)

    score = 0.8 * progress + 0.1 * efficiency + 0.1 * safety

    # Clamp strictly inside (0, 1) — validator rejects exactly 0.0 or 1.0
    return round(min(0.9999, max(0.0001, score)), 4)


# ---------------------------------------------------------------------------
# LiDAR helpers
# ---------------------------------------------------------------------------

def _quadrant_min(bins: list, start: int, end: int) -> float:
    return min(bins[start:end]) if bins[start:end] else 1.0


def lidar_summary(bins: list) -> dict:
    """Collapse 40-bin LiDAR into 4 directional clearances.

    Bin 0 = forward direction, bins increase counter-clockwise:
      front : bins 0-4 and 36-39  (±18° of heading)
      left  : bins 5-14            (45°–126°)
      back  : bins 15-24           (135°–216°)
      right : bins 25-35           (225°–315°)
    """
    if len(bins) < 40:
        return dict(front=1.0, left=1.0, back=1.0, right=1.0)
    front_val = min(_quadrant_min(bins, 0, 5), _quadrant_min(bins, 36, 40))
    left_val  = _quadrant_min(bins, 5, 15)
    back_val  = _quadrant_min(bins, 15, 25)
    right_val = _quadrant_min(bins, 25, 36)
    return dict(front=round(front_val, 2),
                left=round(left_val, 2),
                back=round(back_val, 2),
                right=round(right_val, 2))


def goal_direction_text(angle: float) -> str:
    """Convert normalised goal_angle [-1, 1] to a plain-English bearing."""
    deg = angle * 180.0
    if abs(deg) < 15:
        return "straight ahead"
    side = "left" if deg > 0 else "right"
    return f"{abs(deg):.0f}° to the {side}"


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You control a quadcopter drone navigating to a goal on a 2D map.

    SENSORS you receive each step:
    - LiDAR clearance in 4 directions (0.0 = wall/obstacle right there, 1.0 = open)
    - goal_dist: normalised distance to goal (0.0 = at goal, 1.0 = furthest possible)
    - goal_direction: bearing from drone heading to goal

    COMMAND you must output — a JSON object and nothing else:
    {"vx": <float>, "vy": <float>, "yaw_rate": <float>}

    Ranges and meaning:
    - vx:        0.0 to 1.0  ( 0 = stop, +1 = full forward at 3 m/s)
    - vy:       -1.0 to 1.0  (−1 = strafe right, +1 = strafe left)
    - yaw_rate: -1.0 to 1.0  (−1 = turn right 60°/s, +1 = turn left 60°/s)

    STRATEGY (follow this):
    1. If goal is not straight ahead: yaw toward it (negative yaw_rate to turn right, positive to turn left)
    2. While turning, reduce vx to avoid obstacles
    3. If goal is roughly ahead and front clearance > 0.4: go forward (vx ≈ 0.8)
    4. If front clearance < 0.3: slow down AND steer around the obstacle
    5. Output only the JSON. No explanation.\
""")


def build_user_prompt(step: int, max_steps: int, obs) -> str:
    lidar = lidar_summary(obs.lidar_bins)
    goal_dir = goal_direction_text(obs.goal_angle)
    return (
        f"Step {step}/{max_steps}\n"
        f"LiDAR  front={lidar['front']}  left={lidar['left']}  "
        f"back={lidar['back']}  right={lidar['right']}\n"
        f"Goal   dist={obs.goal_dist:.3f}  direction={goal_dir}\n"
        f"Output JSON:"
    )


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[dict]:
    """Extract {vx, vy, yaw_rate} JSON from model output."""
    text = text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        obj = json.loads(text[start:end])
        vx       = max(-1.0, min(1.0, float(obj.get("vx", 0.0))))
        vy       = max(-1.0, min(1.0, float(obj.get("vy", 0.0))))
        yaw_rate = max(-1.0, min(1.0, float(obj.get("yaw_rate", 0.0))))
        return dict(vx=vx, vy=vy, yaw_rate=yaw_rate)
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


FALLBACK_ACTION = dict(vx=0.5, vy=0.0, yaw_rate=0.0)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(env_client: QuadnavEnv, llm_client: OpenAI, task: dict) -> dict:
    """Run one task episode; returns result dict with score and metrics."""
    task_name = task["name"]
    max_steps = task["max_steps"]

    # ── START ──────────────────────────────────────────────────────────────
    print(f"[START] task={task_name} difficulty={task['difficulty']} max_steps={max_steps}", flush=True)

    result = await env_client.reset(task=task["difficulty"])
    obs    = result.observation
    initial_dist = float(obs.goal_dist)
    print(f"[START] task={task_name} initial_dist={initial_dist:.3f} goal_angle={obs.goal_angle:.3f}", flush=True)

    outcome           = "timeout"
    steps             = 0
    final_dist        = initial_dist
    near_obstacle_steps = 0

    # ── STEP LOOP ──────────────────────────────────────────────────────────
    for step_num in range(1, max_steps + 1):
        if result.done:
            break

        # Count steps where any LiDAR bin is dangerously close
        if obs.lidar_bins and min(obs.lidar_bins) < NEAR_OBSTACLE_THRESHOLD:
            near_obstacle_steps += 1

        user_prompt = build_user_prompt(step_num, max_steps, obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"[STEP] step={step_num} llm_error={exc}", flush=True)
            raw = ""

        action_dict = parse_action(raw) or FALLBACK_ACTION
        action      = QuadnavAction(**action_dict)

        result     = await env_client.step(action)
        obs        = result.observation
        steps      = step_num
        final_dist = float(obs.goal_dist)

        lidar  = lidar_summary(obs.lidar_bins)
        print(f"[STEP] step={step_num} reward={result.reward:+.2f} goal_dist={final_dist:.3f} "
              f"front={lidar['front']} back={lidar['back']} "
              f"left={lidar['left']} right={lidar['right']} done={result.done}", flush=True)

        if result.done:
            break

    # ── END ────────────────────────────────────────────────────────────────
    if result.done:
        state  = await env_client.state()
        outcome = state.outcome

    score = grade(task_name, outcome, initial_dist, final_dist,
                  steps, max_steps, near_obstacle_steps)

    safety_pct = 100.0 * (1.0 - near_obstacle_steps / max(steps, 1))
    print(f"[END] task={task_name} score={score:.4f} outcome={outcome} steps={steps} "
          f"initial_dist={initial_dist:.3f} final_dist={final_dist:.3f} "
          f"near_obstacle_steps={near_obstacle_steps} safety_pct={safety_pct:.0f}", flush=True)

    return dict(task=task_name, score=score, outcome=outcome,
                steps=steps, initial_dist=initial_dist, final_dist=final_dist,
                near_obstacle_steps=near_obstacle_steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"Connecting to Quadnav Environment at {ENV_URL}", flush=True)
    env_client = QuadnavEnv(base_url=ENV_URL)

    results = []
    async with env_client:
        for task in TASKS:
            result = await run_episode(env_client, llm_client, task)
            results.append(result)

    # Summary
    print("\n" + "=" * 55)
    print("QUADNAV SCORES")
    print("=" * 55)
    total = 0.0
    for r in results:
        print(f"  {r['task']:8s}  score={r['score']:.4f}  ({r['outcome']})")
        total += r["score"]
    print("-" * 55)
    print(f"  {'MEAN':8s}  score={total / len(results):.4f}")
    print("=" * 55)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
