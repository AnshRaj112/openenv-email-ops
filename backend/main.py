from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from backend.analytics import analyze_run
from backend.leaderboard import get_leaderboard, save_score
from backend.runner import run_episode
from baseline.run_baseline import run_all_tasks
from env.environment import EmailEnv
from env.models import Action

app = FastAPI()

_current_env: Optional[EmailEnv] = None
_current_task_id: str = "email-triage-easy"


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/tasks")
def tasks():
    # Return tasks directly from `openenv.yaml` for spec alignment.
    with open("openenv.yaml", "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    task_list = spec.get("tasks", [])

    # Minimal action schema derived from the typed Action model.
    action_schema = {
        "type": {"type": "string", "enum": ["respond", "escalate", "archive"], "required": True},
        "email_id": {"type": "string", "required": False, "description": "Defaults to the current email."},
        "content": {
            "type": "string",
            "required": False,
            "required_if": {"type": "respond"},
            "description": "Used to evaluate keyword-based response quality on `respond`.",
        },
        "rationale": {"type": "string", "required": False},
    }
    return {"tasks": task_list, "action_schema": action_schema}


@app.post("/reset")
def reset(task_id: str = "email-triage-easy"):
    global _current_env, _current_task_id
    _current_task_id = task_id
    _current_env = EmailEnv(task_id=task_id)
    obs = _current_env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    if _current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized; call /reset first.")

    obs, reward, done, info = _current_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    if _current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized; call /reset first.")
    return _current_env.state()


@app.get("/baseline")
def baseline():
    # Runs the repo baseline inference over all 3 tasks.
    return run_all_tasks()


@app.post("/grader")
def grader(task_id: str = "email-triage-easy", provider: str = "openai"):
    result = run_episode(task_id=task_id, provider=provider)
    # `run_episode` already returns a deterministic 0.0-1.0 grader score.
    return {"task_id": task_id, "score": result["score"], "steps": result["steps"]}


@app.post("/run")
def run(task_id: str = "email-triage-complex", provider: str = "openai"):
    result = run_episode(task_id=task_id, provider=provider)
    score = result["score"]
    save_score(provider, score)
    analytics = analyze_run(result["steps"])
    return {"score": score, "steps": result["steps"], "analytics": analytics}


@app.get("/leaderboard")
def leaderboard():
    return get_leaderboard()
