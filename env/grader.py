def grade_episode(rewards):
    return sum(r.value for r in rewards) / len(rewards) if rewards else 0