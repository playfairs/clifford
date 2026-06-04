from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from datetime import datetime
import uuid


@dataclass
class Entity:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Relationship:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    target: str = ""
    relation_type: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Fact:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    entities: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
