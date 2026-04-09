import os, sys, json, time, random
from typing import List, Optional
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4o-mini"
BENCHMARK    = "traffic-signal-control"

TASKS = ["easy", "medium", "hard"]
SEEDS = [42, 42, 42]
SUCCESS_SCORE_THRESHOLD = 0.1

TASK_CONFIG = {
    "easy":   {"max_steps": 100, "spawn_prob": 0.2,  "capacity": 10, "min_green": 3, "yellow": 2},
    "medium": {"max_steps": 200, "spawn_prob": 0.4,  "spawn_ew": 0.2, "capacity": 10, "min_green": 4, "yellow": 2},
    "hard":   {"max_steps": 300, "spawn_prob": 0.55, "capacity": 8,  "min_green": 5, "yellow": 2},
}

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

class Lane:
    def __init__(self, cap, prob):
        self.cap = cap; self.prob = prob; self.vehicles = []
    def reset(self): self.vehicles = []
    def update(self, green):
        if len(self.vehicles) < self.cap and random.random() < self.prob:
            self.vehicles.append(0)
        crossed = False
        if green and self.vehicles:
            self.vehicles.pop(0); crossed = True
        self.vehicles = [w+1 for w in self.vehicles]
        return crossed
    @property
    def queue(self): return len(self.vehicles)
    @property
    def avg_wait(self): return sum(self.vehicles)/len(self.vehicles) if self.vehicles else 0.0

class LocalEnv:
    def __init__(self, task_id, seed):
        random.seed(seed)
        cfg = TASK_CONFIG[task_id]
        sp = cfg["spawn_prob"]; sp_ew = cfg.get("spawn_ew", sp); cap = cfg["capacity"]
        self.lanes = {"N": Lane(cap,sp), "S": Lane(cap,sp), "E": Lane(cap,sp_ew), "W": Lane(cap,sp_ew)}
        self.cfg = cfg; self.task_id = task_id
        self.signal = 0; self.time_on = 0; self.yellow = False; self.ycnt = 0
        self.step = 0; self.reward_sum = 0.0; self.reward_count = 0

    def reset(self):
        for l in self.lanes.values(): l.reset()
        self.signal=0; self.time_on=0; self.yellow=False; self.ycnt=0
        self.step=0; self.reward_sum=0.0; self.reward_count=0

    def step_env(self, action):
        if not self.yellow:
            if action != self.signal and self.time_on >= self.cfg["min_green"]:
                self.yellow=True; self.ycnt=0; self.signal=action; self.time_on=0
        if self.yellow:
            self.ycnt += 1
            if self.ycnt >= self.cfg["yellow"]: self.yellow = False
            green_lanes = []
        else:
            green_lanes = ["N","S"] if self.signal == 0 else ["E","W"]
        for name, lane in self.lanes.items():
            lane.update(name in green_lanes)
        tw = sum(l.avg_wait for l in self.lanes.values())
        tq = sum(l.queue for l in self.lanes.values())
        mq = max(l.queue for l in self.lanes.values())
        raw = -1.2*tw - 1.0*tq - 2.0*mq
        reward = min(max((raw + 70) / 78, 0.0), 1.0)
        self.reward_sum += reward; self.reward_count += 1
        self.step += 1; self.time_on += 1
        done = self.step >= self.cfg["max_steps"]
        return reward, done

    def score(self):
        if self.reward_count == 0:
            return 0.0
        return min(max(self.reward_sum / self.reward_count, 0.0), 1.0)

def decide(ns, ew, step, max_steps):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Traffic signal controller. Reply JSON only: {\"signal\": 0 or 1}. 0=NS green, 1=EW green."},
                {"role": "user", "content": f"NS queue={ns}, EW queue={ew}, step={step}/{max_steps}. Choose signal."}
            ],
            max_tokens=20, temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return max(0, min(1, int(json.loads(raw).get("signal", 0 if ns >= ew else 1))))
    except Exception:
        return 0 if ns >= ew else 1

for task_id, seed in zip(TASKS, SEEDS):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = LocalEnv(task_id, seed)
    env.reset()

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    try:
        done = False
        while not done:
            steps_taken += 1
            ns = env.lanes["N"].queue + env.lanes["S"].queue
            ew = env.lanes["E"].queue + env.lanes["W"].queue
            action = decide(ns, ew, env.step, env.cfg["max_steps"])
            reward, done = env.step_env(action)
            rewards.append(reward)
            log_step(step=steps_taken, action=str(action), reward=reward, done=done, error=None)

        final_score = env.score()
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        final_score = env.score()
        success = False
        log_step(step=steps_taken+1, action="0", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    time.sleep(0.5)


def main():
    pass

if __name__ == "__main__":
    pass