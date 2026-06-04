from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import NDArray

from clifford.memory.types import MemoryItem


class SearchableMemory:
    def __init__(self):
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def add_batch(self, items: List[MemoryItem]) -> None:
        self.items.extend(items)

    def search(self, query: str, top_k: int = 5) -> List[tuple[MemoryItem, float]]:
        query_lower = query.lower()
        results = []
        
        for item in self.items:
            content_str = str(item.content).lower()
            metadata_str = str(item.metadata).lower()
            
            score = 0.0
            if query_lower in content_str:
                score += 0.7
            if query_lower in metadata_str:
                score += 0.3
            
            if score > 0:
                results.append((item, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_metadata(self, metadata_key: str, metadata_value: Any) -> List[MemoryItem]:
        results = []
        for item in self.items:
            if metadata_key in item.metadata and item.metadata[metadata_key] == metadata_value:
                results.append(item)
        return results

    def search_by_tag(self, tag: str) -> List[MemoryItem]:
        return [item for item in self.items if tag in item.tags]

    def search_by_importance(self, min_importance: float = 0.0) -> List[MemoryItem]:
        return [item for item in self.items if item.importance >= min_importance]

    def clear(self) -> None:
        self.items.clear()

    def size(self) -> int:
        return len(self.items)
