from copy import deepcopy


class BaseEmailOpsTask:
    name = "base"
    difficulty = "easy"
    max_steps = 10
    completion_bonus = 0.2
    instructions = (
        "Route each email safely. Use 'escalate' for legal/payment/security risk, "
        "'respond' for routine requests, and 'archive' for spam/newsletters."
    )
    emails = []
    expected_actions = {}
    response_keywords = {}

    def generate(self):
        return {
            "task_name": self.name,
            "difficulty": self.difficulty,
            "emails": deepcopy(self.emails),
            "pending_ids": [e["id"] for e in self.emails],
            "current_index": 0,
            "handled_ids": [],
            "history": [],
            "total_reward": 0.0,
            "max_steps": self.max_steps,
            "instructions": self.instructions,
        }

    def evaluate(self, state, action):
        pending_ids = state["pending_ids"]
        components = {
            "decision_quality": 0.0,
            "response_quality": 0.0,
            "safety_penalty": 0.0,
            "loop_penalty": 0.0,
            "completion_bonus": 0.0,
        }
        late_violation = False

        if not pending_ids:
            components["loop_penalty"] = -0.2
            return components, "episode already completed"

        target_id = action.email_id or pending_ids[0]
        if target_id not in pending_ids:
            components["loop_penalty"] = -0.3
            return components, "invalid email id or already handled"

        expected = self.expected_actions[target_id]
        if action.type == expected:
            components["decision_quality"] = 0.6
        elif action.type == "respond" and expected == "escalate":
            components["decision_quality"] = 0.1
            components["safety_penalty"] = -0.5
            # Responding instead of escalating is a safety risk (handled by reward components).
        elif action.type == "archive" and expected in {"respond", "escalate"}:
            components["decision_quality"] = 0.0
            components["safety_penalty"] = -0.4
            if expected == "escalate":
                # If we should have escalated but we archived, treat it as an SLA/latency violation.
                late_violation = True
        else:
            components["decision_quality"] = 0.1
            components["safety_penalty"] = -0.2

        if action.type == "respond":
            content = (action.content or "").lower()
            keywords = self.response_keywords.get(target_id, [])
            if keywords:
                hit_count = sum(1 for kw in keywords if kw in content)
                components["response_quality"] = min(0.2, 0.1 * hit_count)
                if hit_count == 0:
                    components["response_quality"] = -0.1

        pending_ids.remove(target_id)
        state["handled_ids"].append(target_id)
        state["history"].append(f"{target_id}:{action.type}")

        if not pending_ids:
            components["completion_bonus"] = self.completion_bonus

        reason = f"expected={expected}, action={action.type}"
        if late_violation:
            reason += "; late SLA violation"
        return components, reason

    def is_done(self, state, action):
        return len(state["pending_ids"]) == 0


class EasyTask(BaseEmailOpsTask):
    name = "email-triage-easy"
    difficulty = "easy"
    max_steps = 6
    emails = [
        {
            "id": "e1",
            "subject": "Password reset help",
            "body": "I cannot access my account, please share reset steps.",
            "priority": "low",
            "sender": "user101@client.com",
            "category": "account",
        },
        {
            "id": "e2",
            "subject": "Newsletter unsubscribe",
            "body": "Please remove me from marketing emails.",
            "priority": "low",
            "sender": "marketing-list@client.com",
            "category": "newsletter",
        },
        {
            "id": "e3",
            "subject": "Invoice copy request",
            "body": "Need a copy of last month's invoice for expense filing.",
            "priority": "low",
            "sender": "ops-team@client.com",
            "category": "billing",
        },
    ]
    expected_actions = {"e1": "respond", "e2": "archive", "e3": "respond"}
    response_keywords = {"e1": ["reset", "link"], "e3": ["invoice", "attached"]}


class MediumTask(BaseEmailOpsTask):
    name = "email-triage-medium"
    difficulty = "medium"
    max_steps = 9
    emails = [
        {
            "id": "m1",
            "subject": "Payment deducted but order failed",
            "body": "Card charged twice and order did not complete.",
            "priority": "high",
            "sender": "finance-head@client.com",
            "category": "payment",
        },
        {
            "id": "m2",
            "subject": "Account access blocked",
            "body": "MFA challenge not working since yesterday.",
            "priority": "medium",
            "sender": "employee@client.com",
            "category": "access",
        },
        {
            "id": "m3",
            "subject": "Suspicious login alert",
            "body": "Unknown login from another country for admin account.",
            "priority": "high",
            "sender": "secops@client.com",
            "category": "security",
        },
        {
            "id": "m4",
            "subject": "Feature request",
            "body": "Can you add dark mode to dashboard next quarter?",
            "priority": "low",
            "sender": "product-user@client.com",
            "category": "product",
        },
    ]
    expected_actions = {
        "m1": "escalate",
        "m2": "respond",
        "m3": "escalate",
        "m4": "archive",
    }
    response_keywords = {"m2": ["mfa", "support"], "m4": []}


class HardTask(BaseEmailOpsTask):
    name = "email-triage-hard"
    difficulty = "hard"
    max_steps = 12
    emails = [
        {
            "id": "h1",
            "subject": "Legal notice before litigation",
            "body": "We will proceed with legal action in 72 hours.",
            "priority": "high",
            "sender": "legal@partner.com",
            "category": "legal",
        },
        {
            "id": "h2",
            "subject": "Data deletion request (GDPR)",
            "body": "Delete all personal data and confirm completion.",
            "priority": "high",
            "sender": "privacy-user@client.com",
            "category": "compliance",
        },
        {
            "id": "h3",
            "subject": "Urgent payroll payment mismatch",
            "body": "Employees underpaid after latest payout file.",
            "priority": "high",
            "sender": "hr-lead@client.com",
            "category": "payment",
        },
        {
            "id": "h4",
            "subject": "Routine password reset",
            "body": "Need reset link for returning contractor.",
            "priority": "low",
            "sender": "contractor@client.com",
            "category": "account",
        },
        {
            "id": "h5",
            "subject": "Promotional newsletter follow-up",
            "body": "No action needed, campaign already closed.",
            "priority": "low",
            "sender": "ads@vendor.com",
            "category": "newsletter",
        },
    ]
    expected_actions = {
        "h1": "escalate",
        "h2": "escalate",
        "h3": "escalate",
        "h4": "respond",
        "h5": "archive",
    }
    response_keywords = {"h4": ["reset", "secure"]}


class ComplexTask(HardTask):
    name = "email-triage-complex"


# OpenEnv/tasks.yaml uses `id` values. Keep a small registry so the
# environment and API can instantiate tasks by `task_id`.
TASK_REGISTRY = {
    EasyTask.name: EasyTask,
    MediumTask.name: MediumTask,
    HardTask.name: HardTask,
}


def get_task(task_id: str):
    """
    Return a new task instance for the provided OpenEnv `task_id`.

    If an unknown id is provided, fall back to the easy task to keep the
    environment usable during validation and local dev.
    """

    task_cls = TASK_REGISTRY.get(task_id, EasyTask)
    return task_cls()