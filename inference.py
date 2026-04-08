"""
Traffic Signal Control - Baseline Inference Script

Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks and
produces scores in [0.0, 1.0].

Environment variables required:
    API_BASE_URL  - The base URL of the LLM API endpoint
    MODEL_NAME    - Model identifier (e.g. "gpt-4o-mini", "meta-llama/Llama-3-8b-instruct")
    HF_TOKEN      - Your Hugging Face / API key

Environment server (this repo) must be running at http://localhost:7860
(or set ENV_BASE_URL to override).

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="sk-..."
    python inference.py
"""

import os
import sys
import json
import time
import random
from typing import Optional

from openai import OpenAI
import requests

# ============================================================
# CONFIG FROM ENVIRONMENT
# ============================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — API calls may fail.", file=sys.stderr)

TASKS        = ["easy", "medium", "hard"]
SEEDS        = [42, 42, 42]   # deterministic baseline

# ============================================================
# LLM CLIENT
# ============================================================

llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)

# ============================================================
# ENVIRONMENT HTTP HELPERS
# ============================================================

def env_reset(task_id: str, seed: int) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(signal: int, reasoning: str = "") -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"signal": signal, "reasoning": reasoning},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_grade() -> dict:
    r = requests.get(f"{ENV_BASE_URL}/grade", timeout=30)
    r.raise_for_status()
    return r.json()

# ============================================================
# PROMPT BUILDER
# ============================================================

SYSTEM_PROMPT = """You are an expert traffic signal controller AI.
You manage a 4-way intersection with North, South, East, and West lanes.
Your goal is to minimise vehicle wait times and maximise throughput.

At each step you must respond with ONLY a JSON object — no markdown, no explanation outside the JSON:
{
  "signal": <0 or 1>,
  "reasoning": "<one sentence>"
}
signal 0 = give green light to North/South lanes
signal 1 = give green light to East/West lanes"""


def build_user_prompt(obs: dict) -> str:
    n = obs["north"]
    s = obs["south"]
    e = obs["east"]
    w = obs["west"]

    ns_queue = n["queue_length"] + s["queue_length"]
    ew_queue = e["queue_length"] + w["queue_length"]
    ns_wait  = (n["avg_wait_time"] + s["avg_wait_time"]) / 2
    ew_wait  = (e["avg_wait_time"] + w["avg_wait_time"]) / 2

    return (
        f"Step {obs['step']}/{obs['max_steps']} | "
        f"Current signal: {'NS-green' if obs['current_signal']==0 else 'EW-green'} "
        f"(on for {obs['time_on_signal']} steps) | "
        f"Yellow phase: {obs['in_yellow_phase']}\n\n"
        f"Lane queues — N:{n['queue_length']} S:{s['queue_length']} "
        f"E:{e['queue_length']} W:{w['queue_length']}\n"
        f"Avg wait   — NS: {ns_wait:.1f}  EW: {ew_wait:.1f}\n"
        f"NS total queue: {ns_queue}   EW total queue: {ew_queue}\n\n"
        f"Task: {obs['task_description']}\n\n"
        "What signal should be active next? Respond with JSON only."
    )


# ============================================================
# LLM DECISION
# ============================================================

def llm_decide(obs: dict, history: list) -> tuple[int, str]:
    """Call the LLM and parse its signal decision."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep last 6 turns for context (avoid token overflow)
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": build_user_prompt(obs)})

    try:
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=120,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        signal    = int(parsed.get("signal", 0))
        reasoning = str(parsed.get("reasoning", ""))
        signal    = max(0, min(1, signal))   # clamp

        return signal, reasoning

    except Exception as exc:
        # Fallback: pick busier side heuristically
        ns = obs["north"]["queue_length"] + obs["south"]["queue_length"]
        ew = obs["east"]["queue_length"]  + obs["west"]["queue_length"]
        signal = 0 if ns >= ew else 1
        return signal, f"[fallback due to error: {exc}]"


# ============================================================
# RUN ONE TASK
# ============================================================

def run_task(task_id: str, seed: int) -> dict:
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "model": MODEL_NAME,
        "seed": seed,
    }), flush=True)

    obs  = env_reset(task_id, seed)
    done = obs.get("done", False)

    history          = []
    total_reward     = 0.0
    step_count       = 0
    start_time       = time.time()

    while not done:
        signal, reasoning = llm_decide(obs, history)

        # Log the step
        print(json.dumps({
            "type":      "[STEP]",
            "task_id":   task_id,
            "step":      obs["step"],
            "signal":    signal,
            "reasoning": reasoning,
            "ns_queue":  obs["north"]["queue_length"] + obs["south"]["queue_length"],
            "ew_queue":  obs["east"]["queue_length"]  + obs["west"]["queue_length"],
        }), flush=True)

        # Append to history for multi-turn context
        history.append({"role": "user",      "content": build_user_prompt(obs)})
        history.append({"role": "assistant", "content": json.dumps({"signal": signal, "reasoning": reasoning})})

        # Step environment
        result   = env_step(signal, reasoning)
        obs      = result["observation"]
        reward   = result["reward"]
        done     = result["done"]

        total_reward += reward
        step_count   += 1

    # Final grade
    grade_result = env_grade()
    elapsed      = round(time.time() - start_time, 2)

    print(json.dumps({
        "type":          "[END]",
        "task_id":       task_id,
        "score":         grade_result["score"],
        "passed":        grade_result["passed"],
        "threshold":     grade_result["threshold"],
        "steps":         step_count,
        "total_reward":  round(total_reward, 4),
        "elapsed_sec":   elapsed,
    }), flush=True)

    return grade_result


# ============================================================
# MAIN
# ============================================================

def main():
    print(json.dumps({
        "type":     "[START]",
        "run_type": "baseline",
        "model":    MODEL_NAME,
        "api_base": API_BASE_URL,
        "tasks":    TASKS,
    }), flush=True)

    results = {}
    for task_id, seed in zip(TASKS, SEEDS):
        result = run_task(task_id, seed)
        results[task_id] = result["score"]
        time.sleep(0.5)   # brief pause between tasks

    avg_score = sum(results.values()) / len(results)

    print(json.dumps({
        "type":      "[END]",
        "run_type":  "baseline",
        "scores":    results,
        "avg_score": round(avg_score, 4),
        "model":     MODEL_NAME,
    }), flush=True)

    return results


if __name__ == "__main__":
    main()
