from typing import Any, Dict, List, Optional
from collections import deque
import time

from clifford.memory.types import MemoryItem


class ShortTermMemory:
    def __init__(self, capacity: int = 100, decay_time: float = 3600):
        self.capacity = capacity
        self.decay_time = decay_time
        self.items: deque = deque(maxlen=capacity)
        self.timestamps: deque = deque(maxlen=capacity)

    def add(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        item = MemoryItem(content=content, metadata=metadata or {})
        self.items.append(item)
        self.timestamps.append(time.time())
        return item

    def get(self, index: int) -> Optional[MemoryItem]:
        if 0 <= index < len(self.items):
            return list(self.items)[index]
        return None

    def get_recent(self, count: int = 10) -> List[MemoryItem]:
        return list(self.items)[-count:]

    def remove_expired(self) -> int:
        current_time = time.time()
        expired_count = 0
        
        while self.items and (current_time - self.timestamps[0]) > self.decay_time:
            self.items.popleft()
            self.timestamps.popleft()
            expired_count += 1
        
        return expired_count

    def clear(self) -> None:
        self.items.clear()
        self.timestamps.clear()

    def size(self) -> int:
        return len(self.items)

    def get_all(self) -> List[MemoryItem]:
        return list(self.items)
