"""
Traffic Signal Control - FastAPI Server
Exposes OpenEnv-compliant HTTP endpoints: /reset, /step, /state, /grade, /health
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from models import TrafficAction, TrafficObservation, TrafficState
from server.environment import TrafficEnvironment, TASKS

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Traffic Signal Control — OpenEnv",
    description=(
        "An RL environment for training agents to manage traffic signals at a "
        "real-world 4-way intersection. Three tasks of increasing difficulty."
    ),
    version="1.0.0",
)

# Single shared environment instance (one session per container)
env = TrafficEnvironment()


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed:    Optional[int] = None


class StepRequest(BaseModel):
    signal:    int
    reasoning: Optional[str] = None


class StepResponse(BaseModel):
    observation: TrafficObservation
    reward:      float
    done:        bool


class GradeResponse(BaseModel):
    score:       float
    task_id:     str
    step_count:  int
    passed:      bool
    threshold:   float


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions and difficulty."""
    return {
        task_id: {
            "description": cfg["description"],
            "max_steps":   cfg["max_steps"],
            "difficulty":  task_id,
        }
        for task_id, cfg in TASKS.items()
    }


@app.post("/reset", response_model=TrafficObservation)
def reset(req: ResetRequest):
    """
    Start a new episode. Returns the initial observation.
    task_id: one of "easy", "medium", "hard"
    """
    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASKS.keys())}"
        )
    if req.seed is not None:
        import random
        random.seed(req.seed)

    obs = env.reset(task_id=req.task_id)
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Execute one action. Returns (observation, reward, done).
    signal: 0 = keep/switch to NS green, 1 = keep/switch to EW green
    """
    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode."
        )
    action = TrafficAction(signal=req.signal, reasoning=req.reasoning)
    obs, reward, done = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done)


@app.get("/state", response_model=TrafficState)
def state():
    """Return current episode metadata (not the full observation)."""
    return env.state


@app.get("/grade", response_model=GradeResponse)
def grade():
    """
    Compute and return the final score for the current (or just-completed) episode.
    Score is in [0.0, 1.0].
    """
    score     = env.grade()
    task_cfg  = TASKS[env._task_id]
    threshold = task_cfg["success_threshold"]
    return GradeResponse(
        score      = score,
        task_id    = env._task_id,
        step_count = env._step_count,
        passed     = score >= threshold,
        threshold  = threshold,
    )


@app.get("/schema")
def schema():
    """Return JSON schemas for Action, Observation, and State."""
    return {
        "action":      TrafficAction.model_json_schema(),
        "observation": TrafficObservation.model_json_schema(),
        "state":       TrafficState.model_json_schema(),
    }
