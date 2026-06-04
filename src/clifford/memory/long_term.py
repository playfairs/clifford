from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from clifford.memory.types import MemoryItem


class LongTermMemory:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent.parent / "memory" / "long_term.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.items: Dict[str, MemoryItem] = {}
        self._load()

    def add(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        item = MemoryItem(content=content, metadata=metadata or {})
        self.items[item.id] = item
        self._save()
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        item = self.items.get(item_id)
        if item:
            item.access_count += 1
            item.last_accessed = item.last_accessed
            self._save()
        return item

    def remove(self, item_id: str) -> None:
        if item_id in self.items:
            del self.items[item_id]
            self._save()

    def search(self, query: str) -> List[MemoryItem]:
        query_lower = query.lower()
        results = []
        
        for item in self.items.values():
            content_str = str(item.content).lower()
            metadata_str = str(item.metadata).lower()
            
            if query_lower in content_str or query_lower in metadata_str:
                results.append(item)
        
        return sorted(results, key=lambda x: x.importance, reverse=True)

    def get_by_tag(self, tag: str) -> List[MemoryItem]:
        return [item for item in self.items.values() if tag in item.tags]

    def get_by_importance(self, min_importance: float = 0.0) -> List[MemoryItem]:
        return [item for item in self.items.values() if item.importance >= min_importance]

    def clear(self) -> None:
        self.items.clear()
        self._save()

    def size(self) -> int:
        return len(self.items)

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for item_data in data:
                        item = MemoryItem(**item_data)
                        self.items[item.id] = item
            except Exception:
                pass

    def _save(self) -> None:
        data = []
        for item in self.items.values():
            data.append({
                "id": item.id,
                "content": item.content,
                "metadata": item.metadata,
                "timestamp": item.timestamp,
                "importance": item.importance,
                "access_count": item.access_count,
                "last_accessed": item.last_accessed,
                "tags": item.tags,
            })
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
