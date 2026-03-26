class EasyTask:
    def generate(self):
        return {
            "emails": [{
                "id": "1",
                "subject": "Password Reset",
                "body": "I forgot my password",
                "priority": "low"
            }],
            "current": {
                "id": "1",
                "subject": "Password Reset",
                "body": "I forgot my password",
                "priority": "low"
            },
            "history": []
        }

    def evaluate(self, state, action):
        if action.type == "respond":
            if action.content and "reset" in action.content.lower():
                return 1.0, "Correct"
            return 0.5, "Partial"
        return 0.0, "Wrong"

    def is_done(self, state, action):
        return action.type == "respond"


class MediumTask:
    def generate(self):
        return {
            "emails": [{
                "id": "2",
                "subject": "Payment Failed",
                "body": "Money deducted but failed",
                "priority": "high"
            }],
            "current": {
                "id": "2",
                "subject": "Payment Failed",
                "body": "Money deducted but failed",
                "priority": "high"
            },
            "history": []
        }

    def evaluate(self, state, action):
        if action.type == "escalate":
            return 1.0, "Correct"
        if action.type == "respond":
            return 0.3, "Wrong approach"
        return 0.0, "Wrong"

    def is_done(self, state, action):
        return True


class HardTask:
    def generate(self):
        return {
            "emails": [{
                "id": "3",
                "subject": "Legal Notice",
                "body": "I will sue your company",
                "priority": "high"
            }],
            "current": {
                "id": "3",
                "subject": "Legal Notice",
                "body": "I will sue your company",
                "priority": "high"
            },
            "history": []
        }

    def evaluate(self, state, action):
        if action.type == "escalate":
            return 1.0, "Correct"
        if action.type == "respond":
            return 0.2, "Risky"
        return 0.0, "Wrong"

    def is_done(self, state, action):
        return True