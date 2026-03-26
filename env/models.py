from pydantic import BaseModel
from typing import List, Optional

class Email(BaseModel):
    id: str
    subject: str
    body: str
    priority: str

class Observation(BaseModel):
    inbox: List[Email]
    current_email: Optional[Email]
    history: List[str]

class Action(BaseModel):
    type: str
    content: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str