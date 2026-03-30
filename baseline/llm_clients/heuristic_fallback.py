"""When the configured LLM fails, use the same heuristic as `local_client` (not task-oracle actions)."""


def action_from_local_heuristic(prompt: str, parse_action, obs):
    from .local_client import generate as local_generate

    return parse_action(local_generate(prompt), obs)
