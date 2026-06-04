from typing import Any, Dict, List, Optional
from collections import deque
import numpy as np
from numpy.typing import NDArray

from clifford.memory.types import MemoryItem


class WorkingMemory:
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        self.current_focus: Optional[MemoryItem] = None

    def add(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        item = MemoryItem(content=content, metadata=metadata or {})
        self.items.append(item)
        self.current_focus = item
        return item

    def get(self, index: int) -> Optional[MemoryItem]:
        if 0 <= index < len(self.items):
            return list(self.items)[index]
        return None

    def get_current_focus(self) -> Optional[MemoryItem]:
        return self.current_focus

    def set_focus(self, item: MemoryItem) -> None:
        if item in self.items:
            self.current_focus = item
            item.access_count += 1

    def remove(self, item: MemoryItem) -> None:
        if item in self.items:
            self.items.remove(item)
            if self.current_focus == item:
                self.current_focus = None

    def clear(self) -> None:
        self.items.clear()
        self.current_focus = None

    def size(self) -> int:
        return len(self.items)

    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    def get_all(self) -> List[MemoryItem]:
        return list(self.items)
