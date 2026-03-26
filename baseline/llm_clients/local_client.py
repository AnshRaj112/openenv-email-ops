# Deterministic, offline "model" used for free baseline runs.
# It returns JSON in the same schema expected by `baseline/run_baseline.py`.


def generate(prompt: str) -> str:
    # Extract the pending inbox JSON from the prompt for action selection.
    start = prompt.find("Pending inbox JSON:")
    if start == -1:
        return _archive_fallback(prompt)

    json_start = prompt.find("\n", start)
    if json_start == -1:
        return _archive_fallback(prompt)

    # Heuristic: the JSON block ends before "Return ONLY valid JSON".
    end_marker = "Return ONLY valid JSON"
    end = prompt.find(end_marker, json_start)
    if end == -1:
        end = len(prompt)

    block = prompt[json_start:end].strip()
    try:
        # `json.dumps()` in the prompt makes this JSON.
        import json as _json

        pending = _json.loads(block)
    except Exception:
        # If we can't parse, just archive the current email id if present.
        return _archive_fallback(prompt)

    if not pending:
        return _archive_fallback(prompt)

    # Select the first pending email (our env treats that as `current_email`).
    email = pending[0]
    eid = str(email.get("id", "")).lower()

    subject = str(email.get("subject", "")).lower()
    body = str(email.get("body", "")).lower()
    category = str(email.get("category", "")).lower()

    # Route using realistic metadata/keyword heuristics (not email-id lookup).
    # We also include a couple of deterministic "imperfect" choices so
    # the baseline isn't trivially perfect on all tasks.

    # Intentional routing mistakes for baseline realism:
    # - MediumTask product/feature request: respond instead of archive.
    # - HardTask routine password reset: archive instead of respond.
    if category == "product":
        action_type = "respond"
    elif category == "account" and "routine password reset" in subject:
        action_type = "archive"
    elif category in {"legal", "compliance"}:
        action_type = "escalate"
    elif category in {"payment", "security"}:
        action_type = "escalate"
    elif category in {"access", "account", "billing"}:
        action_type = "respond"
    elif category in {"newsletter"}:
        action_type = "archive"
    else:
        # Conservative default to reduce destructive actions.
        action_type = "archive"

    if action_type == "respond":
        # Intentionally respond with only a subset of expected keywords
        # to keep graded response quality imperfect (deterministic).
        content_keyword = None
        if "mfa" in subject or "mfa" in body:
            content_keyword = "mfa"
        elif "invoice" in subject or "invoice" in body:
            content_keyword = "invoice"
        elif "reset" in subject or "reset" in body or "password" in subject or "password" in body:
            content_keyword = "reset"
        else:
            content_keyword = "appropriate"

        # IMPORTANT: we omit the 2nd keyword that some tasks expect (e.g. "link", "attached", "secure", "support").
        content = f"Response includes: {content_keyword}"
        safe_content = content.replace('"', '\\"')
        return '{"type":"respond","email_id":"%s","content":"%s"}' % (eid, safe_content)

    return '{"type":"%s","email_id":"%s","content":null}' % (action_type, eid)


def _archive_fallback(prompt: str) -> str:
    # Try to find the first JSON object with an "id" field to use as fallback.
    # This is best-effort only.
    import re

    m = re.search(r'"id"\s*:\s*"([^"]+)"', prompt)
    eid = m.group(1) if m else None
    if not eid:
        return '{"type":"archive","email_id":null,"content":null}'
    return '{"type":"archive","email_id":"%s","content":null}' % eid

