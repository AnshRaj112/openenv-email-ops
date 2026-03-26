def grade_episode(rewards):
    if not rewards:
        return 0.0
    avg_reward = sum(r.value for r in rewards) / len(rewards)
    # Map reward range [-1, 1] to score range [0, 1].
    return round(max(0.0, min(1.0, (avg_reward + 1.0) / 2.0)), 4)