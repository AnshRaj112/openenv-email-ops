import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from env.environment import EmailEnv
from env.tasks import EasyTask, MediumTask, HardTask
from env.models import Action
from baseline.llm_clients.router import llm_call

def run_task(task):
    env = EmailEnv(task)
    obs = env.reset()
    rewards = []

    for _ in range(5):
        email = obs.current_email

        prompt = f"""
Email:
Subject: {email.subject}
Body: {email.body}

Choose: respond / escalate / archive
"""

        out = llm_call(prompt).lower()

        if "escalate" in out:
            action = Action(type="escalate")
        elif "respond" in out:
            action = Action(type="respond", content=out)
        else:
            action = Action(type="archive")

        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

        if done:
            break

    return sum(r.value for r in rewards) / len(rewards)


if __name__ == "__main__":
    for t in [EasyTask(), MediumTask(), HardTask()]:
        print(t.__class__.__name__, run_task(t))