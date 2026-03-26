def analyze_run(steps):
    total = len(steps)
    risk = sum(1 for s in steps if "dangerous" in s["reason"].lower())
    late = sum(1 for s in steps if "late" in s["reason"].lower())
    return {
        "total_steps": total,
        "risk_actions": risk,
        "sla_violations": late,
        "efficiency": round((total - risk - late) / total, 2) if total else 0,
    }
