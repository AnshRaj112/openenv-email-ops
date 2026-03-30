import json
import os
import sys
from typing import Any, Dict, List, Optional

from baseline.llm_clients.router import llm_call
from baseline.llm_clients.heuristic_fallback import action_from_local_heuristic
from env.environment import EmailEnv
from env.grader import grade_episode
from env.models import Action, Reward


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
    return f"""
You are triaging enterprise support emails.
Instructions: {obs.instructions}
Current step: {obs.step_count}
Remaining emails: {obs.remaining_count}
Pending inbox JSON:
{json.dumps(emails, indent=2)}

Return ONLY valid JSON with this schema:
{{
  "type": "respond|escalate|archive",
  "email_id": "<one pending id>",
  "content": "<required only when type=respond>"
}}
""".strip()


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


def run_episode(task_id: str, provider: str = "local", max_steps_override: Optional[int] = None) -> Dict[str, Any]:
    os.environ["LLM_PROVIDER"] = (provider or "local").strip().lower()

    env = EmailEnv(task_id=task_id)
    obs = env.reset()

    steps: List[Dict[str, Any]] = []
    rewards: List[Reward] = []

    max_steps = env.state_data.get("max_steps", 10)  # task provides this in generate()
    if max_steps_override is not None:
        max_steps = int(max_steps_override)

    llm_failure_warned = False
    for _ in range(max_steps):
        prompt = _build_prompt(obs)
        try:
            out = llm_call(prompt)
            action = _parse_action(out, obs)
        except Exception as exc:
            if not llm_failure_warned:
                print(
                    f"[runner] LLM call failed ({exc!r}); using local heuristic fallback.",
                    file=sys.stderr,
                )
                llm_failure_warned = True
            action = action_from_local_heuristic(prompt, _parse_action, obs)

        obs, reward, done, info = env.step(action)

        email_dump = obs.current_email.model_dump() if obs.current_email else None
        steps.append(
            {
                "task": info.get("task"),
                "difficulty": info.get("difficulty"),
                "handled": info.get("handled"),
                "remaining": info.get("remaining"),
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                # Also include the next email to make debugging traces easier.
                "next_current_email": email_dump,
            }
        )
        rewards.append(reward)

        if done:
            break

    score = grade_episode(rewards)
    return {
        "task_id": task_id,
        "task_name": env.task.name,
        "difficulty": env.task.difficulty,
        "provider": provider,
        "score": score,
        "steps": steps,
    }
