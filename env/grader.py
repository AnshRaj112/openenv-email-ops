import re


def _parse_action_from_reason(reason: str) -> str:
    # reward.reason is deterministically built in `BaseEmailOpsTask.evaluate()`:
    #   "expected=<x>, action=<type>" + optional "; late SLA violation"
    m = re.search(r"action=([a-zA-Z_]+)", reason or "")
    return m.group(1).lower() if m else ""


def grade_episode(rewards):
    """
    Deterministic episode grader (0.0-1.0).

    Success criteria (clear + reproducible):
    - completion: all emails handled (completion_bonus present)
    - correctness: every step took the expected routing action type
    - response safety: no negative response_quality on `respond` steps

    Uses dense reward components for partial credit when the agent is close.
    """

    if not rewards:
        return 0.0

    n = len(rewards)
    completion = any(r.components.get("completion_bonus", 0.0) > 0.0 for r in rewards)

    # decision_quality is set by `BaseEmailOpsTask.evaluate()`:
    # - correct routing: 0.6
    # - wrong routing (but somewhat aligned): 0.1
    # - wrong & highly misaligned: 0.0
    # Give partial credit proportionally instead of a hard threshold.
    decision_steps = [r for r in rewards]
    correctness_ratio = (
        sum(
            max(0.0, min(1.0, (r.components.get("decision_quality", 0.0) / 0.6)))
            for r in decision_steps
        )
        / n
    )

    respond_steps = []
    response_quality_clamped = []
    response_negative = False

    for r in rewards:
        action_type = _parse_action_from_reason(r.reason or "")
        if action_type == "respond":
            q = r.components.get("response_quality", 0.0)
            respond_steps.append(r)
            if q < 0:
                response_negative = True
            response_quality_clamped.append(max(0.0, q))

    # response_quality is in [-0.1, 0.2] when keywords are configured.
    if response_quality_clamped:
        response_quality_ratio = sum(response_quality_clamped) / len(response_quality_clamped) / 0.2
    else:
        # If the task has no keyword checks for respond steps, treat as neutral.
        response_quality_ratio = 1.0

    # Safety penalties are negative values (e.g., -0.5 for wrong escalation).
    penalties = []
    for r in rewards:
        sp = r.components.get("safety_penalty", 0.0)
        lp = r.components.get("loop_penalty", 0.0)
        penalties.append(max(0.0, -sp) + max(0.0, -lp))

    avg_penalty = sum(penalties) / n
    # Normalize typical max penalty to [0, 1] without being too sensitive.
    penalty_factor = max(0.0, min(1.0, avg_penalty / 1.0))

    # Only award a perfect score when the agent is not only correct and complete,
    # but also achieves perfect response keyword quality on all `respond` steps.
    success = (
        completion
        and correctness_ratio >= 1.0 - 1e-9
        and (response_quality_ratio >= 1.0 - 1e-9)
        and not response_negative
    )
    if success:
        return 1.0

    # Weighted partial credit across the whole trajectory.
    score = (
        0.55 * correctness_ratio
        + 0.25 * (1.0 if completion else 0.0)
        + 0.15 * max(0.0, min(1.0, response_quality_ratio))
        - 0.20 * penalty_factor
    )

    return round(max(0.0, min(1.0, score)), 4)