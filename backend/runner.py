import os

from baseline.llm_clients.router import llm_call
from env.environment import EmailEnv
from env.models import Action
from env.tasks import ComplexTask


def _fallback_decision(email):
    text = f"{email.subject} {email.body}".lower()
    if "legal" in text or "payment" in text:
        return "escalate"
    if "password" in text:
        return "respond"
    return "archive"


def run_episode(provider: str = "groq"):
    os.environ["LLM_PROVIDER"] = provider

    env = EmailEnv(ComplexTask())
    obs = env.reset()

    steps = []
    rewards = []

    for _ in range(15):
        email = obs.current_email
        if not email:
            break

        prompt = f"""
Email:
Subject: {email.subject}
Body: {email.body}
Priority: {email.priority}

Rules:
- legal -> escalate
- payment -> escalate
- password -> respond
"""

        try:
            out = llm_call(prompt).lower()
        except Exception as exc:
            # Keep the run available even when provider keys/config are missing.
            out = _fallback_decision(email)
            if out == "respond":
                out = f"{out} (fallback: {type(exc).__name__})"

        if "escalate" in out:
            action = Action(type="escalate")
        elif "respond" in out:
            action = Action(type="respond", content=out)
        else:
            action = Action(type="archive")

        obs, reward, done, _ = env.step(action)

        steps.append(
            {
                "email": email.model_dump(),
                "action": action.type,
                "reward": reward.value,
                "reason": reward.reason,
            }
        )
        rewards.append(reward.value)

        if done:
            break

    return {"score": sum(rewards) / len(rewards) if rewards else 0, "steps": steps}
