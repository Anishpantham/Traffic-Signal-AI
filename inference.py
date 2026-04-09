import os, sys, json, time, random
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "no-key"))
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = ["easy", "medium", "hard"]
SEEDS = [42, 42, 42]

TASK_CONFIG = {
    "easy":   {"max_steps": 100, "spawn_prob": 0.2,  "capacity": 10, "min_green": 3, "yellow": 2},
    "medium": {"max_steps": 200, "spawn_prob": 0.4,  "spawn_ew": 0.2, "capacity": 10, "min_green": 4, "yellow": 2},
    "hard":   {"max_steps": 300, "spawn_prob": 0.55, "capacity": 8,  "min_green": 5, "yellow": 2},
}

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
    @property
    def density(self): return self.queue / self.cap

class LocalEnv:
    def __init__(self, task_id, seed):
        random.seed(seed)
        cfg = TASK_CONFIG[task_id]
        sp = cfg["spawn_prob"]; sp_ew = cfg.get("spawn_ew", sp); cap = cfg["capacity"]
        self.lanes = {"N": Lane(cap,sp), "S": Lane(cap,sp), "E": Lane(cap,sp_ew), "W": Lane(cap,sp_ew)}
        self.cfg = cfg; self.task_id = task_id
        self.signal = 0; self.time_on = 0; self.yellow = False; self.ycnt = 0
        self.step = 0; self.throughput = 0; self.cum_wait = 0.0
        self.reward_sum = 0.0; self.reward_count = 0

    def reset(self):
        for l in self.lanes.values(): l.reset()
        self.signal=0; self.time_on=0; self.yellow=False; self.ycnt=0
        self.step=0; self.throughput=0; self.cum_wait=0.0
        self.reward_sum=0.0; self.reward_count=0
        return self._obs(0.5, False)

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
            crossed = lane.update(name in green_lanes)
            if crossed: self.throughput += 1
        tw = sum(l.avg_wait for l in self.lanes.values())
        self.cum_wait += tw
        tq = sum(l.queue for l in self.lanes.values())
        mq = max(l.queue for l in self.lanes.values())
        raw = -1.2*tw - 1.0*tq - 2.0*mq
        reward = max(0.001, min(0.999, (raw + 70) / 78))
        self.reward_sum += reward
        self.reward_count += 1
        self.step += 1; self.time_on += 1
        done = self.step >= self.cfg["max_steps"]
        return self._obs(reward, done), reward, done

    def grade(self):
        if self.reward_count == 0:
            return 0.5
        avg = self.reward_sum / self.reward_count
        return round(min(0.999, max(0.001, avg)), 4)

    def _obs(self, reward, done):
        r = round(min(0.999, max(0.001, float(reward))), 4)
        return {
            "step": self.step, "max_steps": self.cfg["max_steps"],
            "current_signal": self.signal, "time_on_signal": self.time_on,
            "in_yellow_phase": self.yellow, "done": done, "reward": r,
            "north": {"queue_length": self.lanes["N"].queue, "avg_wait_time": round(self.lanes["N"].avg_wait,2), "density": round(self.lanes["N"].density,4)},
            "south": {"queue_length": self.lanes["S"].queue, "avg_wait_time": round(self.lanes["S"].avg_wait,2), "density": round(self.lanes["S"].density,4)},
            "east":  {"queue_length": self.lanes["E"].queue, "avg_wait_time": round(self.lanes["E"].avg_wait,2), "density": round(self.lanes["E"].density,4)},
            "west":  {"queue_length": self.lanes["W"].queue, "avg_wait_time": round(self.lanes["W"].avg_wait,2), "density": round(self.lanes["W"].density,4)},
        }

def decide(obs):
    ns = obs["north"]["queue_length"] + obs["south"]["queue_length"]
    ew = obs["east"]["queue_length"]  + obs["west"]["queue_length"]
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Traffic signal controller. Reply JSON only: {\"signal\": 0 or 1}. 0=NS green, 1=EW green."},
                {"role": "user", "content": f"NS queue={ns}, EW queue={ew}, step={obs['step']}/{obs['max_steps']}. Choose signal."}
            ],
            max_tokens=20, temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        signal = max(0, min(1, int(json.loads(raw).get("signal", 0 if ns >= ew else 1))))
        return signal, f"LLM: NS={ns} EW={ew}"
    except Exception:
        signal = 0 if ns >= ew else 1
        return signal, f"heuristic: NS={ns} EW={ew}"

print(f"[START] run_type=baseline model={MODEL_NAME}", flush=True)
print(json.dumps({"type": "[START]", "run_type": "baseline", "model": MODEL_NAME, "tasks": TASKS}), flush=True)

results = {}
for task_id, seed in zip(TASKS, SEEDS):
    print(f"[START] task={task_id} model={MODEL_NAME} seed={seed}", flush=True)
    print(json.dumps({"type": "[START]", "task_id": task_id, "model": MODEL_NAME, "seed": seed}), flush=True)

    env = LocalEnv(task_id, seed)
    obs = env.reset()
    done = obs["done"]
    total_reward = 0.0; step_count = 0; start_time = time.time()

    while not done:
        signal, reasoning = decide(obs)
        reward = round(min(0.999, max(0.001, float(obs.get("reward") or 0.5))), 4)
        print(f"[STEP] task={task_id} step={obs['step']} signal={signal} reward={reward}", flush=True)
        print(json.dumps({"type": "[STEP]", "task_id": task_id, "step": obs["step"],
                          "signal": signal, "reasoning": reasoning, "reward": reward,
                          "ns_queue": obs["north"]["queue_length"]+obs["south"]["queue_length"],
                          "ew_queue": obs["east"]["queue_length"]+obs["west"]["queue_length"]}), flush=True)
        obs, reward, done = env.step_env(signal)
        total_reward += reward; step_count += 1

    score = round(min(0.999, max(0.001, env.grade())), 4)
    passed = score > 0.001
    results[task_id] = score
    print(f"[END] task={task_id} score={score} passed={passed} steps={step_count}", flush=True)
    print(json.dumps({"type": "[END]", "task_id": task_id, "score": score, "passed": passed,
                      "steps": step_count, "elapsed_sec": round(time.time()-start_time, 2)}), flush=True)
    time.sleep(0.5)

avg_score = round(min(0.999, max(0.001, sum(results.values())/len(results))), 4)
print(f"[END] run_type=baseline avg_score={avg_score} model={MODEL_NAME}", flush=True)
print(json.dumps({"type": "[END]", "run_type": "baseline", "scores": results, "avg_score": avg_score, "model": MODEL_NAME}), flush=True)

def main():
    pass

if __name__ == "__main__":
    pass