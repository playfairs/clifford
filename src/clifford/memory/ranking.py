from typing import List, Optional, Callable, Dict
from datetime import datetime, timedelta

from clifford.memory.types import MemoryItem


class MemoryRanker:
    @staticmethod
    def rank_by_importance(items: List[MemoryItem]) -> List[MemoryItem]:
        return sorted(items, key=lambda x: x.importance, reverse=True)

    @staticmethod
    def rank_by_access_count(items: List[MemoryItem]) -> List[MemoryItem]:
        return sorted(items, key=lambda x: x.access_count, reverse=True)

    @staticmethod
    def rank_by_recency(items: List[MemoryItem]) -> List[MemoryItem]:
        return sorted(items, key=lambda x: x.timestamp, reverse=True)

    @staticmethod
    def rank_by_last_accessed(items: List[MemoryItem]) -> List[MemoryItem]:
        return sorted(items, key=lambda x: x.last_accessed, reverse=True)

    @staticmethod
    def rank_by_decay(items: List[MemoryItem], decay_rate: float = 0.1) -> List[MemoryItem]:
        current_time = datetime.now()
        
        def calculate_score(item: MemoryItem) -> float:
            try:
                item_time = datetime.fromisoformat(item.timestamp)
                time_diff = (current_time - item_time).total_seconds()
                decay = decay_rate ** (time_diff / 3600)
                return item.importance * decay
            except:
                return item.importance
        
        return sorted(items, key=calculate_score, reverse=True)

    @staticmethod
    def rank_by_custom(items: List[MemoryItem], scoring_func: Callable[[MemoryItem], float]) -> List[MemoryItem]:
        return sorted(items, key=scoring_func, reverse=True)

    @staticmethod
    def rank_combined(items: List[MemoryItem], weights: Optional[Dict[str, float]] = None) -> List[MemoryItem]:
        if weights is None:
            weights = {
                "importance": 0.4,
                "access_count": 0.3,
                "recency": 0.2,
                "last_accessed": 0.1
            }
        
        current_time = datetime.now()
        
        def calculate_combined_score(item: MemoryItem) -> float:
            score = 0.0
            
            if "importance" in weights:
                score += weights["importance"] * item.importance
            
            if "access_count" in weights:
                max_access = max((i.access_count for i in items), default=1)
                score += weights["access_count"] * (item.access_count / max_access if max_access > 0 else 0)
            
            if "recency" in weights:
                try:
                    item_time = datetime.fromisoformat(item.timestamp)
                    time_diff = (current_time - item_time).total_seconds()
                    max_diff = max(( (current_time - datetime.fromisoformat(i.timestamp)).total_seconds() for i in items if i.timestamp), default=1)
                    score += weights["recency"] * (1 - time_diff / max_diff if max_diff > 0 else 0)
                except:
                    pass
            
            if "last_accessed" in weights:
                try:
                    last_access_time = datetime.fromisoformat(item.last_accessed)
                    time_diff = (current_time - last_access_time).total_seconds()
                    max_diff = max(( (current_time - datetime.fromisoformat(i.last_accessed)).total_seconds() for i in items if i.last_accessed), default=1)
                    score += weights["last_accessed"] * (1 - time_diff / max_diff if max_diff > 0 else 0)
                except:
                    pass
            
            return score
        
        return sorted(items, key=calculate_combined_score, reverse=True)
