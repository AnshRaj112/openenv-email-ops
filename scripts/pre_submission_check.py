import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi.testclient import TestClient
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from env.environment import EmailEnv
from env.models import Action, Observation, Reward
from env.tasks import EasyTask, MediumTask, HardTask
from env.grader import grade_episode
from baseline.run_baseline import run_all_tasks


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_openenv_yaml():
    with open("openenv.yaml", "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    _assert(isinstance(spec, dict), "openenv.yaml must parse as a dict")
    for key in ["name", "version", "description", "entrypoint", "observation_model", "action_model", "reward_model", "tasks"]:
        _assert(key in spec, f"openenv.yaml missing key: {key}")

    tasks = spec.get("tasks", [])
    _assert(isinstance(tasks, list) and len(tasks) >= 3, "openenv.yaml must define at least 3 tasks")


def check_typed_models_and_env():
    env = EmailEnv(task=EasyTask())
    obs = env.reset()
    _assert(isinstance(obs, Observation), "reset() must return Observation")

    action = Action(type="archive")
    obs2, reward, done, info = env.step(action)
    _assert(isinstance(obs2, Observation), "step() must return Observation")
    _assert(isinstance(reward, Reward), "step() must return Reward")
    _assert(isinstance(done, bool), "step() done must be bool")
    _assert(isinstance(info, dict), "step() info must be dict")

    # Also validate grader accepts reward objects.
    _ = grade_episode([reward])


def check_http_endpoints_with_testclient():
    client = TestClient(app)

    r = client.get("/")
    _assert(r.status_code == 200, "/ should return 200")

    r = client.get("/tasks")
    _assert(r.status_code == 200, "/tasks should return 200")
    payload = r.json()
    _assert("tasks" in payload and "action_schema" in payload, "/tasks response shape unexpected")

    r = client.post("/reset", params={"task_id": "email-triage-easy"})
    _assert(r.status_code == 200, "/reset should return 200")
    obs_payload = r.json()
    _assert("inbox" in obs_payload, "/reset response must look like Observation")

    # Submit one step with an archive action.
    r = client.post(
        "/step",
        json={"type": "archive", "email_id": None, "content": None, "rationale": None},
    )
    _assert(r.status_code == 200, "/step should return 200")
    payload = r.json()
    _assert("observation" in payload and "reward" in payload and "done" in payload, "/step response shape unexpected")

    r = client.get("/state")
    _assert(r.status_code == 200, "/state should return 200")

    # Avoid external API calls.
    os.environ["LLM_PROVIDER"] = "local"
    r = client.get("/baseline")
    _assert(r.status_code == 200, "/baseline should return 200")
    b = r.json()
    _assert("task_scores" in b and "overall" in b, "/baseline response shape unexpected")

    r = client.post("/grader", params={"task_id": "email-triage-easy", "provider": "local"})
    _assert(r.status_code == 200, "/grader should return 200")
    g = r.json()
    _assert(0.0 <= float(g["score"]) <= 1.0, "/grader score must be in [0,1]")


def main():
    # Keep validation self-contained; don't call remote models.
    os.environ.setdefault("LLM_PROVIDER", "local")

    # If the `openenv` CLI is present in the execution environment, run it.
    # Otherwise, fall back to our local smoke checks.
    try:
        completed = subprocess.run(
            ["openenv", "validate"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "openenv validate failed.\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n"
            )
    except FileNotFoundError:
        print("openenv CLI not found; skipping `openenv validate` (local smoke checks only).")

    check_openenv_yaml()
    check_typed_models_and_env()
    check_http_endpoints_with_testclient()

    # Basic smoke test for baseline scoring.
    res = run_all_tasks()
    _assert("overall" in res and "task_scores" in res, "baseline run_all_tasks() result shape unexpected")

    print("Pre-submission checks: OK")


if __name__ == "__main__":
    main()

