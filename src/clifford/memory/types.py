from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from datetime import datetime
import uuid


@dataclass
class MemoryItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)


@dataclass
class EpisodicMemory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5


@dataclass
class SemanticMemory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    concept: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    confidence: float = 1.0
