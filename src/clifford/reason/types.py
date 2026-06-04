from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from datetime import datetime
import uuid


@dataclass
class ReasoningStep:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thought: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Decision:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = ""
    reasoning: List[ReasoningStep] = field(default_factory=list)
    confidence: float = 0.5
    alternatives: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    outcome: Optional[str] = None


@dataclass
class Plan:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    steps: List[str] = field(default_factory=list)
    current_step: int = 0
    status: str = "pending"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
