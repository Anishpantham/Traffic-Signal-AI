"""
Traffic Signal Control - Server Environment Logic
"""

import uuid
import random
from typing import Dict, Tuple

from models import TrafficAction, TrafficObservation, TrafficState, LaneState


TASKS: Dict[str, dict] = {
    "easy": {
        "description": (
            "Low-traffic intersection. Keep average wait time below 5 steps "
            "across all 4 lanes over 100 simulation steps."
        ),
        "max_steps": 100,
        "spawn_prob": 0.2,
        "lane_capacity": 10,
        "min_green_time": 3,
        "yellow_time": 2,
        "target_avg_wait": 5.0,
        "success_threshold": 0.70,
    },
    "medium": {
        "description": (
            "High-traffic intersection with unbalanced flow (NS twice as busy). "
            "Minimise total queue and prevent any lane exceeding 8 vehicles over 200 steps."
        ),
        "max_steps": 200,
        "spawn_prob": 0.4,
        "spawn_prob_ew": 0.2,
        "lane_capacity": 10,
        "min_green_time": 4,
        "yellow_time": 2,
        "target_avg_wait": 8.0,
        "success_threshold": 0.60,
    },
    "hard": {
        "description": (
            "Rush-hour surge: all lanes very busy, capacity constrained to 8. "
            "Achieve total throughput >= 180 vehicles in 300 steps without "
            "any lane queue exceeding 7 for more than 10 consecutive steps."
        ),
        "max_steps": 300,
        "spawn_prob": 0.55,
        "lane_capacity": 8,
        "min_green_time": 5,
        "yellow_time": 2,
        "target_throughput": 180,
        "max_overflow_steps": 10,
        "overflow_queue": 7,
        "success_threshold": 0.55,
    },
}


class _Vehicle:
    def __init__(self):
        self.wait_time = 0
        self.crossed   = False


class _Lane:
    def __init__(self, capacity: int, spawn_prob: float):
        self.capacity   = capacity
        self.spawn_prob = spawn_prob
        self.vehicles: list = []

    def reset(self):
        self.vehicles = []

    def spawn(self):
        if len(self.vehicles) < self.capacity and random.random() < self.spawn_prob:
            self.vehicles.append(_Vehicle())

    def update(self, is_green: bool) -> bool:
        self.spawn()
        crossed = False
        if is_green and self.vehicles:
            v = self.vehicles.pop(0)
            v.crossed = True
            crossed   = True
        for v in self.vehicles:
            v.wait_time += 1
        return crossed

    @property
    def queue_length(self) -> int:
        return len(self.vehicles)

    @property
    def avg_wait(self) -> float:
        if not self.vehicles:
            return 0.0
        return sum(v.wait_time for v in self.vehicles) / len(self.vehicles)

    @property
    def density(self) -> float:
        return self.queue_length / self.capacity


class TrafficEnvironment:

    def __init__(self):
        self._episode_id     = str(uuid.uuid4())
        self._task_id        = "easy"
        self._cfg            = TASKS["easy"]
        self._step_count     = 0
        self._total_reward   = 0.0
        self._done           = False
        self._signal         = 0
        self._time_on_signal = 0
        self._in_yellow      = False
        self._yellow_counter = 0
        self._lanes: Dict[str, _Lane] = {}
        self._total_throughput   = 0
        self._overflow_streak: Dict[str, int] = {"N": 0, "S": 0, "E": 0, "W": 0}
        self._max_overflow_steps = 0
        self._cumulative_wait    = 0.0
        self._step_rewards: list = []

    def reset(self, task_id: str = "easy") -> TrafficObservation:
        self._task_id        = task_id
        self._cfg            = TASKS[task_id]
        self._episode_id     = str(uuid.uuid4())
        self._step_count     = 0
        self._total_reward   = 0.0
        self._done           = False
        self._signal         = 0
        self._time_on_signal = 0
        self._in_yellow      = False
        self._yellow_counter = 0
        self._total_throughput   = 0
        self._overflow_streak    = {"N": 0, "S": 0, "E": 0, "W": 0}
        self._max_overflow_steps = 0
        self._cumulative_wait    = 0.0
        self._step_rewards       = []

        sp    = self._cfg["spawn_prob"]
        sp_ew = self._cfg.get("spawn_prob_ew", sp)
        cap   = self._cfg["lane_capacity"]

        self._lanes = {
            "N": _Lane(cap, sp),
            "S": _Lane(cap, sp),
            "E": _Lane(cap, sp_ew),
            "W": _Lane(cap, sp_ew),
        }

        return self._observe(reward=0.5, done=False)

    def step(self, action: TrafficAction) -> Tuple[TrafficObservation, float, bool]:
        if self._done:
            raise RuntimeError("Episode is done — call reset() first.")

        requested = action.signal
        safe      = self._safe_signal(requested)

        if not self._in_yellow:
            if safe != self._signal:
                self._in_yellow      = True
                self._yellow_counter = 0
                self._signal         = safe
                self._time_on_signal = 0

        if self._in_yellow:
            self._yellow_counter += 1
            if self._yellow_counter >= self._cfg["yellow_time"]:
                self._in_yellow = False
            green_lanes = []
        else:
            green_lanes = ["N", "S"] if self._signal == 0 else ["E", "W"]

        step_throughput = 0
        for name, lane in self._lanes.items():
            crossed = lane.update(is_green=(name in green_lanes))
            if crossed:
                step_throughput += 1

        self._total_throughput += step_throughput

        for name, lane in self._lanes.items():
            overflow_q = self._cfg.get("overflow_queue", 7)
            if lane.queue_length > overflow_q:
                self._overflow_streak[name] += 1
                self._max_overflow_steps = max(
                    self._max_overflow_steps, self._overflow_streak[name]
                )
            else:
                self._overflow_streak[name] = 0

        total_wait = sum(l.avg_wait for l in self._lanes.values())
        self._cumulative_wait += total_wait

        raw_reward  = self._compute_raw_reward(step_throughput, total_wait)
        norm_reward = self._normalise_reward(raw_reward)
        self._total_reward += norm_reward
        self._step_rewards.append(norm_reward)

        self._step_count     += 1
        self._time_on_signal += 1
        done = self._step_count >= self._cfg["max_steps"]
        self._done = done

        obs = self._observe(reward=norm_reward, done=done)
        return obs, norm_reward, done

    @property
    def state(self) -> TrafficState:
        return TrafficState(
            episode_id   = self._episode_id,
            step_count   = self._step_count,
            task_id      = self._task_id,
            total_reward = self._total_reward,
            is_done      = self._done,
        )

    def grade(self) -> float:
        if self._step_count == 0:
            return 0.001
        if self._task_id == "easy":
            return self._grade_easy()
        elif self._task_id == "medium":
            return self._grade_medium()
        else:
            return self._grade_hard()

    def _grade_easy(self) -> float:
        avg_wait_per_step = self._cumulative_wait / max(1, self._step_count) / 4
        target = self._cfg["target_avg_wait"]
        score = 1.0 - avg_wait_per_step / (2 * target)
        return round(min(0.999, max(0.001, score)), 4)

    def _grade_medium(self) -> float:
        avg_wait_per_step = self._cumulative_wait / max(1, self._step_count) / 4
        target = self._cfg["target_avg_wait"]
        wait_score = 1.0 - avg_wait_per_step / (2 * target)
        overflow_penalty = min(0.998, self._max_overflow_steps / self._cfg["max_steps"])
        overflow_score   = 1.0 - overflow_penalty
        score = 0.5 * wait_score + 0.5 * overflow_score
        return round(min(0.999, max(0.001, score)), 4)

    def _grade_hard(self) -> float:
        target_tp = self._cfg["target_throughput"]
        throughput_score = self._total_throughput / target_tp
        max_allowed = self._cfg["max_overflow_steps"]
        overflow_ratio   = min(0.998, self._max_overflow_steps / max(1, max_allowed))
        overflow_penalty = overflow_ratio * 0.4
        score = throughput_score - overflow_penalty
        return round(min(0.999, max(0.001, score)), 4)

    def _safe_signal(self, requested: int) -> int:
        if self._in_yellow:
            return self._signal
        if requested != self._signal and self._time_on_signal < self._cfg["min_green_time"]:
            return self._signal
        return requested

    def _compute_raw_reward(self, throughput: int, total_wait: float) -> float:
        total_queue = sum(l.queue_length for l in self._lanes.values())
        max_queue   = max(l.queue_length  for l in self._lanes.values())
        switch_pen  = 1 if self._in_yellow else 0
        return (
            -1.2 * total_wait
            - 1.0 * total_queue
            - 2.0 * max_queue
            + 2.0 * throughput
            - 0.5 * switch_pen
        )

    def _normalise_reward(self, raw: float) -> float:
        WORST = -70.0
        BEST  =   8.0
        clipped = max(WORST, min(BEST, raw))
        return round(min(0.999, max(0.001, (clipped - WORST) / (BEST - WORST))), 4)

    def _observe(self, reward, done) -> TrafficObservation:
        def lane_state(lane: _Lane) -> LaneState:
            return LaneState(
                queue_length  = lane.queue_length,
                avg_wait_time = round(lane.avg_wait, 2),
                density       = round(lane.density,  4),
            )

        return TrafficObservation(
            north            = lane_state(self._lanes.get("N", _Lane(10, 0))),
            south            = lane_state(self._lanes.get("S", _Lane(10, 0))),
            east             = lane_state(self._lanes.get("E", _Lane(10, 0))),
            west             = lane_state(self._lanes.get("W", _Lane(10, 0))),
            current_signal   = self._signal,
            time_on_signal   = self._time_on_signal,
            in_yellow_phase  = self._in_yellow,
            step             = self._step_count,
            max_steps        = self._cfg["max_steps"],
            task_id          = self._task_id,
            task_description = self._cfg["description"],
            done             = done,
            reward           = round(min(0.999, max(0.001, float(reward))), 4),
            info             = {
                "total_throughput": self._total_throughput,
                "cumulative_wait":  round(self._cumulative_wait, 2),
            },
        )