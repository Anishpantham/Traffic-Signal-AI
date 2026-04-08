import os, sys, json, time, random
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "no-key"))
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "")

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

    def reset(self):
        for l in self.lanes.values(): l.reset()
        self.signal=0; self.time_on=0; self.yellow=False; self.ycnt=0
        self.step=0; self.throughput=0; self.cum_wait=0.0
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
        self.step += 1; self.time_on += 1
        done = self.step >= self.cfg["max_steps"]
        return self._obs(reward, done), reward, done

    def grade(self):
        avg_wait = self.cum_wait / max(1, self.step) / 4
        if self.task_id == "easy":
            return round(min(0.999, max(0.001, 1.0 - avg_wait / 10.0)), 4)
        elif self.task_id == "medium":
            return round(min(0.999, max(0.001, 1.0 - avg_wait / 16.0)), 4)
        else:
            return round(min(0.999, max(0.001, self.throughput / 180.0)), 4)

    def _obs(self, reward, done):
        return {
            "step": self.step, "max_steps": self.cfg["max_steps"],
            "current_signal": self.signal, "time_on_signal": self.time_on,
            "in_yellow_phase": self.yellow, "done": done, "reward": reward,
            "north": {"queue_length": self.lanes["N"].queue, "avg_wait_time": self.lanes["N"].avg_wait, "density": self.lanes["N"].density},
            "south": {"queue_length": self.lanes["S"].queue, "avg_wait_time": self.lanes["S"].avg_wait, "density": self.lanes["S"].density},
            "east":  {"queue_length": self.lanes["E"].queue, "avg_wait_time": self.lanes["E"].avg_wait, "density": self.lanes["E"].density},
            "west":  {"queue_length": self.lanes["W"].queue, "avg_wait_time": self.lanes["W"].avg_wait, "density": self.lanes["W"].density},
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
        return signal, f"LLM: NS={ns} EW={ew} -> {'NS' if signal==0 else 'EW'}"
    except Exception as e:
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
        reward = float(obs.get("reward") or 0.5)
        reward = min(0.999, max(0.001, reward))
        print(f"[STEP] task={task_id} step={obs['step']} signal={signal} reward={round(reward,4)}", flush=True)
        print(json.dumps({"type": "[STEP]", "task_id": task_id, "step": obs["step"],
                          "signal": signal, "reasoning": reasoning, "reward": round(reward, 4),
                          "ns_queue": obs["north"]["queue_length"]+obs["south"]["queue_length"],
                          "ew_queue": obs["east"]["queue_length"]+obs["west"]["queue_length"]}), flush=True)
        obs, reward, done = env.step_env(signal)
        total_reward += reward; step_count += 1

    score = min(0.999, max(0.001, env.grade()))
    passed = score >= {"easy": 0.70, "medium": 0.60, "hard": 0.55}[task_id]
    results[task_id] = score
    print(f"[END] task={task_id} score={score} passed={passed} steps={step_count}", flush=True)
    print(json.dumps({"type": "[END]", "task_id": task_id, "score": score, "passed": passed,
                      "steps": step_count, "total_reward": round(min(0.999, max(0.001, total_reward/max(1,step_count))), 4),
                      "elapsed_sec": round(time.time()-start_time,2)}), flush=True)
    time.sleep(0.5)

avg_score = round(min(0.999, max(0.001, sum(results.values())/len(results))), 4)
print(f"[END] run_type=baseline avg_score={avg_score} model={MODEL_NAME}", flush=True)
print(json.dumps({"type": "[END]", "run_type": "baseline", "scores": results, "avg_score": avg_score, "model": MODEL_NAME}), flush=True)

def main():
    pass

if __name__ == "__main__":
    pass