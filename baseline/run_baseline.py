import sys
from pathlib import Path
import json

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from env.environment import EmailEnv
from env.tasks import EasyTask, MediumTask, HardTask
from env.models import Action
from env.grader import grade_episode
from baseline.llm_clients.router import llm_call

def _build_prompt(obs):
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
"""


def _parse_action(raw_text, obs):
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


def run_task(task):
    env = EmailEnv(task)
    obs = env.reset()
    rewards = []

    for _ in range(task.max_steps):
        prompt = _build_prompt(obs)
        out = llm_call(prompt)
        action = _parse_action(out, obs)

        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

        if done:
            break

    return grade_episode(rewards)


if __name__ == "__main__":
    task_scores = {}
    for task in [EasyTask(), MediumTask(), HardTask()]:
        score = run_task(task)
        task_scores[task.name] = score
        print(f"{task.name}: {score:.4f}")
    overall = sum(task_scores.values()) / len(task_scores)
    print(f"overall: {overall:.4f}")