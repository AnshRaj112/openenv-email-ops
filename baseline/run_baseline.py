import sys
from pathlib import Path
import json
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load repo `.env` from project root so API keys work when cwd is not the repo.
load_dotenv(PROJECT_ROOT / ".env")

from env.environment import EmailEnv
from env.tasks import EasyTask, MediumTask, HardTask
from env.models import Action
from env.grader import grade_episode
from baseline.llm_clients.router import llm_call
from baseline.llm_clients.heuristic_fallback import action_from_local_heuristic

# Provider selection: Groq, Gemini, or local only.
groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
provider = os.getenv("LLM_PROVIDER", "").strip().lower()

if not provider:
    if groq_key:
        provider = "groq"
    elif gemini_key:
        provider = "gemini"
    else:
        provider = "local"

os.environ["LLM_PROVIDER"] = provider

# Log the first LLM failure once per process (all tasks may hit the same misconfiguration).
_LLM_FAILURE_LOGGED = False


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
    global _LLM_FAILURE_LOGGED

    for _ in range(task.max_steps):
        prompt = _build_prompt(obs)
        try:
            out = llm_call(prompt)
            action = _parse_action(out, obs)
        except Exception as exc:
            # Do not use expected_actions here — that oracle always scores ~1.0 on the grader.
            # Fall back to the same heuristic as the `local` provider.
            if not _LLM_FAILURE_LOGGED:
                print(
                    f"[baseline] LLM call failed ({exc!r}); "
                    "using local heuristic fallback (scores will match LLM_PROVIDER=local).",
                    file=sys.stderr,
                )
                _LLM_FAILURE_LOGGED = True
            action = action_from_local_heuristic(prompt, _parse_action, obs)

        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

        if done:
            break

    return grade_episode(rewards)


def run_all_tasks():
    task_scores = {}
    for task in [EasyTask(), MediumTask(), HardTask()]:
        score = run_task(task)
        task_scores[task.name] = score
    overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return {"task_scores": task_scores, "overall": overall}


if __name__ == "__main__":
    res = run_all_tasks()
    for name, score in res["task_scores"].items():
        print(f"{name}: {score:.4f}")
    print(f"overall: {res['overall']:.4f}")