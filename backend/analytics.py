def analyze_run(steps):
    total = len(steps)
    def _reason(s):
        if isinstance(s, dict):
            if "reason" in s and isinstance(s["reason"], str):
                return s["reason"]
            reward = s.get("reward") if isinstance(s.get("reward"), dict) else {}
            return reward.get("reason", "") if isinstance(reward.get("reason", ""), str) else ""
        return ""

    risk = sum(1 for s in steps if "dangerous" in _reason(s).lower())
    late = sum(1 for s in steps if "late" in _reason(s).lower())
    return {
        "total_steps": total,
        "risk_actions": risk,
        "sla_violations": late,
        "efficiency": round((total - risk - late) / total, 2) if total else 0,
    }
