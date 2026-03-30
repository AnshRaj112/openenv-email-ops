from typing import Tuple, Dict, Any, Optional, Union
from .models import Observation, Action, Reward
from .tasks import EasyTask, get_task

class EmailEnv:
    def __init__(self, task: Optional[Union[object, str]] = None, task_id: Optional[str] = None):
        """
        OpenEnv-compatible environment wrapper.

        Supports constructing with:
        - an instantiated task object (must implement generate/evaluate/is_done)
        - a `task_id` string from `openenv.yaml`
        - no args (defaults to the easy task; useful for validation tools)
        """

        if task_id is not None and task is not None:
            raise ValueError("Provide either `task` or `task_id`, not both.")

        if task is None:
            if task_id is not None:
                self.task = get_task(task_id)
            else:
                self.task = EasyTask()
        else:
            # Accept either a task object or a task_id string for convenience.
            if isinstance(task, str):
                self.task = get_task(task)
            else:
                self.task = task
        self.state_data = None
        self.steps = 0
        # Deterministic per-env episode seed so environment instances are reproducible.
        self.episode_count = 0

    def reset(self) -> Observation:
        self.episode_count += 1
        # Tasks may use the seed to deterministically vary content/labels per episode.
        self.state_data = self.task.generate(seed=self.episode_count)
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