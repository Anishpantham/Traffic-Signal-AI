"""
Traffic Signal Control - OpenEnv Typed Models
Action, Observation, and State Pydantic definitions.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================
# ACTION
# ============================================================

class TrafficAction(BaseModel):
    signal: int = Field(..., ge=0, le=1, description="0=NS green, 1=EW green")
    reasoning: Optional[str] = Field(None, description="Agent's optional reasoning")


# ============================================================
# OBSERVATION
# ============================================================

class LaneState(BaseModel):
    queue_length: int = Field(..., description="Number of vehicles waiting")
    avg_wait_time: float = Field(..., description="Average wait time in steps")
    density: float = Field(..., description="Queue as fraction of lane capacity (0.0-1.0)")


class TrafficObservation(BaseModel):
    north: LaneState
    south: LaneState
    east:  LaneState
    west:  LaneState

    current_signal: int   = Field(..., description="Active signal phase: 0=NS, 1=EW")
    time_on_signal: int   = Field(..., description="Steps spent on current phase")
    in_yellow_phase: bool = Field(..., description="True during yellow/transition phase")

    step: int             = Field(..., description="Current step in episode")
    max_steps: int        = Field(..., description="Total episode length")

    task_id: str          = Field(..., description="Active task identifier")
    task_description: str = Field(..., description="Human-readable task goal")

    done:   bool          = Field(False)
    reward: float         = Field(0.5, description="Normalised reward (0, 1)")
    info:   Dict[str, float] = Field(default_factory=dict)


# ============================================================
# STATE
# ============================================================

class TrafficState(BaseModel):
    episode_id: str
    step_count: int = 0
    task_id: str    = "easy"
    total_reward: float = 0.0
    is_done: bool   = False