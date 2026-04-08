"""
Traffic Signal Control - Python Client
Use this in your RL training or inference code.

Usage:
    from client import TrafficSignalEnv

    env = TrafficSignalEnv(base_url="http://localhost:7860")

    obs = env.reset(task_id="easy")
    while not obs.done:
        obs, reward, done = env.step(signal=0)
        print(reward)

    score = env.grade()
    print(f"Final score: {score.score:.3f}")
"""

import requests
from typing import Optional, Tuple

from models import TrafficAction, TrafficObservation, TrafficState


class TrafficSignalEnv:
    """
    HTTP client for the Traffic Signal Control OpenEnv server.
    Compatible with Hugging Face Spaces deployments.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ----------------------------------------------------------
    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ----------------------------------------------------------
    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> TrafficObservation:
        """Start a new episode. Returns the initial observation."""
        r = self._post("/reset", {"task_id": task_id, "seed": seed})
        return TrafficObservation(**r)

    # ----------------------------------------------------------
    def step(self, signal: int, reasoning: Optional[str] = None) -> Tuple[TrafficObservation, float, bool]:
        """
        Execute one step.
        Returns (observation, reward, done).
        """
        r = self._post("/step", {"signal": signal, "reasoning": reasoning})
        obs    = TrafficObservation(**r["observation"])
        reward = r["reward"]
        done   = r["done"]
        return obs, reward, done

    # ----------------------------------------------------------
    def state(self) -> TrafficState:
        """Return current episode metadata."""
        r = self._get("/state")
        return TrafficState(**r)

    # ----------------------------------------------------------
    def grade(self) -> dict:
        """Return the final score for the current episode."""
        return self._get("/grade")

    # ----------------------------------------------------------
    def list_tasks(self) -> dict:
        return self._get("/tasks")

    # ----------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        r = self._session.post(f"{self.base_url}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self._session.get(f"{self.base_url}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
