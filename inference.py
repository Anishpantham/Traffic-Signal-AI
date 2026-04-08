"""
Traffic Signal Control - Baseline Inference Script

When run by the validator (no API key), uses a built-in heuristic agent.
When run locally with API credentials, uses the LLM agent.

Environment variables (all optional):
    API_BASE_URL  - LLM API endpoint
    MODEL_NAME    - Model identifier  
    HF_TOKEN      - API key
    ENV_BASE_URL  - Environment server URL
"""

import os
import sys
import json
import time

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://anishpantham-traffic-signal-control.hf.space")

TASKS = ["easy", "medium", "hard"]
SEEDS = [42, 42, 42]

def env_reset(task_id, seed):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=60)
    r.raise_for_status()
    return r.json()

def env_step(signal, reasoning=""):
    r = requests.post(f"{ENV_BASE_URL}/step", json={"signal": signal, "reasoning": reasoning}, timeout=60)
    r.raise_for_status()
    return r.json()

def env_grade():
    r = requests.get(f"{ENV_BASE_URL}/grade", timeout=60)
    r.raise_for_status()
    return r.json()

def heuristic_decide(obs):
    ns = obs["north"]["queue_length"] + obs["south"]["queue_length"]
    ew = obs["east"]["queue_length"]  + obs["west"]["queue_length"]
    signal = 0 if ns >= ew else 1
    return signal, f"NS={ns} EW={ew}, giving green to {'NS' if signal==0 else 'EW'}"

def llm_decide(obs, history):
    if not HF_TOKEN:
        return heuristic_decide(obs)
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        n=obs["north"]; s=obs["south"]; e=obs["east"]; w=obs["west"]
        user_msg = (f"Step {obs['step']}/{obs['max_steps']} | Signal: {'NS' if obs['current_signal']==0 else 'EW'}-green\n"
                    f"Queues N:{n['queue_length']} S:{s['queue_length']} E:{e['queue_length']} W:{w['queue_length']}\n"
                    f"Task: {obs['task_description']}\nRespond JSON only: {{\"signal\": 0 or 1, \"reasoning\": \"...\"}}")
        messages = [{"role":"system","content":"Control traffic signals. Respond ONLY with JSON: {\"signal\":0 or 1,\"reasoning\":\"...\"}"}]
        messages += history[-4:] + [{"role":"user","content":user_msg}]
        resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=100, temperature=0.0)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        return max(0,min(1,int(parsed.get("signal",0)))), str(parsed.get("reasoning",""))
    except Exception:
        return heuristic_decide(obs)

def run_task(task_id, seed):
    print(json.dumps({"type":"[START]","task_id":task_id,"model":MODEL_NAME if HF_TOKEN else "heuristic","seed":seed}), flush=True)
    obs = env_reset(task_id, seed)
    done = obs.get("done", False)
    history=[]; total_reward=0.0; step_count=0; start_time=time.time()

    while not done:
        signal, reasoning = llm_decide(obs, history)
        print(json.dumps({
            "type":"[STEP]","task_id":task_id,"step":obs["step"],
            "signal":signal,"reasoning":reasoning,
            "ns_queue":obs["north"]["queue_length"]+obs["south"]["queue_length"],
            "ew_queue":obs["east"]["queue_length"]+obs["west"]["queue_length"],
            "reward":obs.get("reward",0),
        }), flush=True)
        history.append({"role":"user","content":str(obs["step"])})
        history.append({"role":"assistant","content":json.dumps({"signal":signal})})
        result = env_step(signal, reasoning)
        obs=result["observation"]; reward=result["reward"]; done=result["done"]
        total_reward+=reward; step_count+=1

    grade_result = env_grade()
    print(json.dumps({
        "type":"[END]","task_id":task_id,"score":grade_result["score"],
        "passed":grade_result["passed"],"threshold":grade_result["threshold"],
        "steps":step_count,"total_reward":round(total_reward,4),
        "elapsed_sec":round(time.time()-start_time,2),
    }), flush=True)
    return grade_result

def main():
    print(json.dumps({"type":"[START]","run_type":"baseline","model":MODEL_NAME if HF_TOKEN else "heuristic","api_base":API_BASE_URL,"env_url":ENV_BASE_URL,"tasks":TASKS}), flush=True)
    results = {}
    for task_id, seed in zip(TASKS, SEEDS):
        try:
            results[task_id] = run_task(task_id, seed)["score"]
        except Exception as e:
            print(json.dumps({"type":"[END]","task_id":task_id,"error":str(e),"score":0.0,"passed":False}), flush=True)
            results[task_id] = 0.0
        time.sleep(0.5)
    avg_score = sum(results.values())/len(results)
    print(json.dumps({"type":"[END]","run_type":"baseline","scores":results,"avg_score":round(avg_score,4),"model":MODEL_NAME if HF_TOKEN else "heuristic"}), flush=True)
    return results

if __name__ == "__main__":
    main()