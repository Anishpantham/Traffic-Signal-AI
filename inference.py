import os, sys, json, time, requests

sys.stdout.reconfigure(line_buffering=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://anishpantham-traffic-signal-control.hf.space")

TASKS = ["easy", "medium", "hard"]
SEEDS = [42, 42, 42]

print(json.dumps({"type": "[START]", "run_type": "baseline", "model": MODEL_NAME, "tasks": TASKS}), flush=True)

def env_reset(task_id, seed):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=120)
    r.raise_for_status()
    return r.json()

def env_step(signal, reasoning=""):
    r = requests.post(f"{ENV_BASE_URL}/step", json={"signal": signal, "reasoning": reasoning}, timeout=120)
    r.raise_for_status()
    return r.json()

def env_grade():
    r = requests.get(f"{ENV_BASE_URL}/grade", timeout=120)
    r.raise_for_status()
    return r.json()

def decide(obs):
    ns = obs["north"]["queue_length"] + obs["south"]["queue_length"]
    ew = obs["east"]["queue_length"]  + obs["west"]["queue_length"]
    signal = 0 if ns >= ew else 1
    return signal, f"NS={ns} EW={ew} giving {'NS' if signal==0 else 'EW'} green"

def run_task(task_id, seed):
    print(json.dumps({"type": "[START]", "task_id": task_id, "model": MODEL_NAME, "seed": seed}), flush=True)
    try:
        obs  = env_reset(task_id, seed)
        done = obs.get("done", False)
        total_reward = 0.0
        step_count   = 0
        start_time   = time.time()

        while not done:
            signal, reasoning = decide(obs)
            print(json.dumps({
                "type":      "[STEP]",
                "task_id":   task_id,
                "step":      obs["step"],
                "signal":    signal,
                "reasoning": reasoning,
                "reward":    obs.get("reward", 0),
                "ns_queue":  obs["north"]["queue_length"] + obs["south"]["queue_length"],
                "ew_queue":  obs["east"]["queue_length"]  + obs["west"]["queue_length"],
            }), flush=True)

            result       = env_step(signal, reasoning)
            obs          = result["observation"]
            reward       = result["reward"]
            done         = result["done"]
            total_reward += reward
            step_count   += 1

        grade_result = env_grade()
        print(json.dumps({
            "type":         "[END]",
            "task_id":      task_id,
            "score":        grade_result["score"],
            "passed":       grade_result["passed"],
            "threshold":    grade_result["threshold"],
            "steps":        step_count,
            "total_reward": round(total_reward, 4),
            "elapsed_sec":  round(time.time() - start_time, 2),
        }), flush=True)
        return grade_result["score"]

    except Exception as e:
        print(json.dumps({"type": "[END]", "task_id": task_id, "score": 0.0, "passed": False, "error": str(e)}), flush=True)
        return 0.0

results = {}
for task_id, seed in zip(TASKS, SEEDS):
    results[task_id] = run_task(task_id, seed)
    time.sleep(1)

avg_score = sum(results.values()) / len(results)
print(json.dumps({"type": "[END]", "run_type": "baseline", "scores": results, "avg_score": round(avg_score, 4), "model": MODEL_NAME}), flush=True)

def main():
    pass

if __name__ == "__main__":
    pass