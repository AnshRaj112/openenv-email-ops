from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

class Email(BaseModel):
    id: str
    subject: str
    body: str
    priority: str
    sender: str = "customer@company.com"
    category: str = "general"

class Observation(BaseModel):
    inbox: List[Email]
    current_email: Optional[Email]
    history: List[str]
    step_count: int = 0
    remaining_count: int = 0
    instructions: str = ""

class Action(BaseModel):
    type: Literal["respond", "escalate", "archive"]
    email_id: Optional[str] = None
    content: Optional[str] = None
    rationale: Optional[str] = None

class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reason: str
    components: Dict[str, float] = Field(default_factory=dict)