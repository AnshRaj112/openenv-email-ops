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

        reward_value, reason = self.task.evaluate(self.state_data, action)

        done = self.task.is_done(self.state_data, action) or self.steps > 10

        return self._get_obs(), Reward(value=reward_value, reason=reason), done, {}

    def state(self):
        return self.state_data

    def _get_obs(self):
        return Observation(
            inbox=self.state_data["emails"],
            current_email=self.state_data["current"],
            history=self.state_data["history"]
        )