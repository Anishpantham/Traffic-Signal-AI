---
title: Traffic Signal Control
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - traffic
  - reinforcement-learning
  - real-world
license: mit
---

# 🚦 Traffic Signal Control — OpenEnv Environment

A real-world **traffic signal control** environment where an LLM agent manages
a 4-way intersection by deciding which direction gets the green light at each
simulation timestep. The agent must balance competing demands: minimise vehicle
wait times, prevent lane overflow, and maximise throughput.

---

## Why Traffic Signal Control?

Traffic signal optimisation is a genuine engineering problem faced by every city
on earth. Poor signal timing wastes millions of vehicle-hours per year and
increases emissions. Modern intersections collect real-time sensor data — making
this a perfect testbed for language-model agents that must reason over structured
state, respect hard constraints (minimum green time, yellow phases), and adapt
their strategy as traffic patterns shift.

---

## Environment Description

The environment simulates a single 4-way intersection with:
- **4 lanes**: North (N), South (S), East (E), West (W)
- **Signal phases**: NS-green (0) or EW-green (1)
- **Safety constraints**: minimum green time before switching, yellow transition phase
- **Vehicle dynamics**: stochastic spawning, queue accumulation, per-vehicle wait tracking

At each step the agent observes the full intersection state and chooses a signal phase.
The environment enforces the minimum green time constraint automatically.

---

## Action Space

    TrafficAction
    ├── signal    int  [0, 1]       0=NS-green  1=EW-green
    └── reasoning str  (optional)   Agent's explanation (logged, not used in sim)

## Observation Space

    TrafficObservation
    ├── north / south / east / west   LaneState
    │   ├── queue_length   int        vehicles waiting
    │   ├── avg_wait_time  float      mean wait in steps
    │   └── density        float      queue / capacity  [0, 1]
    ├── current_signal     int        active phase
    ├── time_on_signal     int        steps on current phase
    ├── in_yellow_phase    bool       transition in progress
    ├── step               int        current timestep
    ├── max_steps          int        episode length
    ├── task_id            str        active task
    ├── task_description   str        goal text
    ├── done               bool
    └── reward             float      normalised reward [0, 1]

---

## Tasks

| ID | Difficulty | Steps | Traffic | Goal | Pass threshold |
|----|-----------|-------|---------|------|----------------|
| `easy`   | 🟢 Easy   | 100 | Low, balanced (spawn 0.20) | Keep avg wait < 5 steps | 0.70 |
| `medium` | 🟡 Medium | 200 | High, unbalanced NS (0.40/0.20) | Minimise queue + prevent overflow | 0.60 |
| `hard`   | 🔴 Hard   | 300 | Rush-hour surge (spawn 0.55) | Hit throughput target, avoid sustained overflow | 0.55 |

### Reward Function

    raw = -1.2 × total_wait
        - 1.0 × total_queue
        - 2.0 × max_queue
        + 2.0 × throughput
        - 0.5 × switch_penalty

Normalised to **[0.0, 1.0]** at every timestep.

### Final Graders

| Task | Grader formula |
|------|---------------|
| `easy` | `score = max(0, 1 - avg_wait / (2 × target_wait))` |
| `medium` | `0.5 × wait_score + 0.5 × (1 - overflow_fraction)` |
| `hard` | `min(1, throughput/target) - overflow_penalty (up to 0.40)` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start new episode. Body: `{"task_id": "easy", "seed": 42}` |
| POST | `/step` | Execute action. Body: `{"signal": 0, "reasoning": "..."}` |
| GET | `/state` | Episode metadata |
| GET | `/grade` | Final score [0, 1] |
| GET | `/tasks` | List all tasks |
| GET | `/schema` | JSON schemas for Action/Observation/State |
| GET | `/health` | Health check |

---

## Quickstart

### Local (Python)

    git clone https://huggingface.co/spaces/<AnishPantham>/traffic-signal-control
    cd traffic-signal-control
    pip install -r requirements.txt
    uvicorn server.app:app --host 0.0.0.0 --port 7860

### Docker

    docker build -t traffic-signal-env .
    docker run -p 7860:7860 traffic-signal-env

### Run the baseline inference script

    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama-3.1-8b-instant"
    export HF_TOKEN="your_groq_key_here"
    python3 inference.py

### Use the Python client

    from client import TrafficSignalEnv

    env = TrafficSignalEnv(base_url="http://localhost:7860")
    obs = env.reset(task_id="medium", seed=42)

    while not obs.done:
        signal = 0 if obs.north.queue_length + obs.south.queue_length >= \
                      obs.east.queue_length  + obs.west.queue_length else 1
        obs, reward, done = env.step(signal=signal)

    grade = env.grade()
    print(f"Score: {grade['score']:.3f}  Passed: {grade['passed']}")

---

## Baseline Scores

Scores produced by `llama-3.1-8b-instant` (via Groq) with seed=42, temperature=0:

| Task | Score | Passed |
|------|-------|--------|
| easy | 0.547 | ❌ |
| medium | 0.632 | ✅ |
| hard | 0.720 | ✅ |
| **avg** | **0.633** | |

---

## Project Structure

    traffic-signal-env/
    ├── models.py              # Pydantic: TrafficAction, TrafficObservation, TrafficState
    ├── client.py              # Python HTTP client
    ├── inference.py           # Baseline LLM inference script
    ├── openenv.yaml           # OpenEnv manifest
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    └── server/
        ├── __init__.py
        ├── app.py             # FastAPI server
        └── environment.py     # Simulation logic + 3 graded tasks

---

## License

MIT