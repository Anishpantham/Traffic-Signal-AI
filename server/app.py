"""
Traffic Signal Control - FastAPI Server
Exposes OpenEnv-compliant HTTP endpoints: /reset, /step, /state, /grade, /health
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from models import TrafficAction, TrafficObservation, TrafficState
from server.environment import TrafficEnvironment, TASKS

app = FastAPI(
    title="Traffic Signal Control — OpenEnv",
    version="1.0.0",
)

env = TrafficEnvironment()


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "description": cfg["description"],
            "max_steps":   cfg["max_steps"],
            "difficulty":  task_id,
        }
        for task_id, cfg in TASKS.items()
    }


@app.post("/reset", response_model=TrafficObservation)
async def reset(request: Request):
    """
    Start a new episode. Body is fully optional.
    Accepts POST with no body, empty body, or {"task_id": "easy", "seed": 42}
    """
    task_id = "easy"
    seed    = None

    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = body.get("task_id", "easy") or "easy"
            seed    = body.get("seed", None)
    except Exception:
        pass

    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}"
        )
    if seed is not None:
        import random
        random.seed(seed)

    obs = env.reset(task_id=task_id)
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
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
    return env.state


@app.get("/grade", response_model=GradeResponse)
def grade():
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
    return {
        "action":      TrafficAction.model_json_schema(),
        "observation": TrafficObservation.model_json_schema(),
        "state":       TrafficState.model_json_schema(),
    }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()