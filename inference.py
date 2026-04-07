from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from env.environment import EmailEnv
from env.grader import grade_episode
from env.models import Action
from env.tasks import EasyTask, HardTask, MediumTask

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
if not API_KEY:
    raise RuntimeError("Missing API key: set API_KEY or HF_TOKEN.")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "openenv-email-ops"


def _build_prompt(obs) -> str:
    emails = [
        {
            "id": e.id,
            "subject": e.subject,
            "body": e.body,
            "priority": e.priority,
            "category": e.category,
        }
        for e in obs.inbox
    ]
    return (
        "You are triaging enterprise support emails.\n"
        f"Instructions: {obs.instructions}\n"
        f"Current step: {obs.step_count}\n"
        f"Remaining emails: {obs.remaining_count}\n"
        f"Pending inbox JSON:\n{json.dumps(emails, separators=(',', ':'))}\n\n"
        "Return ONLY valid JSON with this schema:\n"
        '{"type":"respond|escalate|archive","email_id":"<id>","content":"<required only for respond>"}'
    )


def _parse_action(raw_text: Optional[str], obs) -> Action:
    pending_ids = {item.id for item in obs.inbox}
    fallback_id = obs.current_email.id if obs.current_email else None
    fallback = Action(type="archive", email_id=fallback_id, content=None)
    if not raw_text:
        return fallback
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        data = json.loads(raw_text[start : end + 1])
    except Exception:
        return fallback

    action_type = str(data.get("type", "archive")).lower()
    if action_type not in {"respond", "escalate", "archive"}:
        action_type = "archive"

    email_id = data.get("email_id") or fallback_id
    if email_id not in pending_ids:
        email_id = fallback_id

    content = data.get("content")
    if action_type != "respond":
        content = None

    return Action(type=action_type, email_id=email_id, content=content)


def _log_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _log_step(step: int, action: Action, reward: float, done: bool, error: Optional[str]) -> None:
    action_str = f"{action.type}:{action.email_id or 'none'}"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_str}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _run_single_task(task, client: OpenAI) -> float:
    env = EmailEnv(task)
    obs = env.reset()
    rewards = []
    done = False
    steps_taken = 0
    _log_start(task.name)

    try:
        for step in range(1, task.max_steps + 1):
            raw_text = ""
            step_error: Optional[str] = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Return only JSON."},
                        {"role": "user", "content": _build_prompt(obs)},
                    ],
                    temperature=0.0,
                    max_tokens=180,
                )
                raw_text = (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                step_error = str(exc)
            action = _parse_action(raw_text, obs)

            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            steps_taken = step
            _log_step(step=step, action=action, reward=float(reward.value), done=done, error=step_error)
            if done:
                break
    finally:
        score = max(0.0, min(1.0, float(grade_episode(rewards)) if rewards else 0.0))
        _log_end(success=done, steps=steps_taken, score=score, rewards=[float(r.value) for r in rewards])

    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in [EasyTask(), MediumTask(), HardTask()]:
        _run_single_task(task, client)


if __name__ == "__main__":
    main()

