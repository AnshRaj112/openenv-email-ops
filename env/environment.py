from typing import Tuple, Dict, Any
from .models import Observation, Action, Reward

class EmailEnv:
    def __init__(self, task):
        self.task = task
        self.state_data = None
        self.steps = 0

    def reset(self) -> Observation:
        self.state_data = self.task.generate()
        self.steps = 0
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.steps += 1

        components, reason = self.task.evaluate(self.state_data, action)
        reward_value = sum(components.values())
        self.state_data["total_reward"] += reward_value

        done = self.task.is_done(self.state_data, action) or self.steps >= self.state_data["max_steps"]
        if self.steps >= self.state_data["max_steps"] and not self.task.is_done(self.state_data, action):
            reward_value -= 0.2
            components["loop_penalty"] = components.get("loop_penalty", 0.0) - 0.2
            reason = f"{reason}; max-steps penalty applied"

        reward_value = max(-1.0, min(1.0, reward_value))
        info = {
            "task": self.state_data["task_name"],
            "difficulty": self.state_data["difficulty"],
            "handled": len(self.state_data["handled_ids"]),
            "remaining": len(self.state_data["pending_ids"]),
        }
        return self._get_obs(), Reward(value=reward_value, reason=reason, components=components), done, info

    def state(self):
        return self.state_data

    def _get_obs(self):
        pending = self.state_data["pending_ids"]
        all_emails = {item["id"]: item for item in self.state_data["emails"]}
        inbox = [all_emails[eid] for eid in pending]
        current = inbox[0] if inbox else None
        return Observation(
            inbox=inbox,
            current_email=current,
            history=self.state_data["history"],
            step_count=self.steps,
            remaining_count=len(pending),
            instructions=self.state_data["instructions"],
        )