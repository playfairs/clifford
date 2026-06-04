from typing import List, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray
from collections import Counter

from clifford.memory.types import MemoryItem


class MemorySummarizer:
    @staticmethod
    def summarize_text(text: str, max_length: int = 100) -> str:
        words = text.split()
        if len(words) <= max_length:
            return text
        return " ".join(words[:max_length]) + "..."

    @staticmethod
    def summarize_items(items: List[MemoryItem], max_items: int = 5) -> Dict[str, Any]:
        if not items:
            return {
                "count": 0,
                "items": [],
                "summary": "No items to summarize"
            }
        
        top_items = sorted(items, key=lambda x: x.importance, reverse=True)[:max_items]
        
        return {
            "count": len(items),
            "top_items": top_items,
            "summary": f"Top {min(max_items, len(items))} items out of {len(items)} total",
            "total_importance": sum(item.importance for item in items),
            "avg_importance": np.mean([item.importance for item in items]),
        }

    @staticmethod
    def summarize_by_tags(items: List[MemoryItem]) -> Dict[str, List[MemoryItem]]:
        tag_groups = {}
        for item in items:
            for tag in item.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(item)
        return tag_groups

    @staticmethod
    def summarize_by_importance(items: List[MemoryItem], bins: int = 3) -> Dict[str, List[MemoryItem]]:
        if not items:
            return {}
        
        importances = [item.importance for item in items]
        min_imp = min(importances)
        max_imp = max(importances)
        
        if min_imp == max_imp:
            return {"medium": items}
        
        bin_size = (max_imp - min_imp) / bins
        groups = {}
        
        for item in items:
            bin_index = int((item.importance - min_imp) / bin_size)
            bin_index = min(bin_index, bins - 1)
            
            if bin_index == 0:
                group_name = "low"
            elif bin_index == bins - 1:
                group_name = "high"
            else:
                group_name = "medium"
            
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(item)
        
        return groups

    @staticmethod
    def generate_summary(items: List[MemoryItem]) -> str:
        if not items:
            return "No memories available"
        
        total_items = len(items)
        total_importance = sum(item.importance for item in items)
        avg_importance = total_importance / total_items if total_items > 0 else 0
        
        tag_counter = Counter()
        for item in items:
            for tag in item.tags:
                tag_counter[tag] += 1
        
        top_tags = tag_counter.most_common(5)
        
        summary_parts = [
            f"Total memories: {total_items}",
            f"Average importance: {avg_importance:.2f}",
            f"Top tags: {', '.join(tag for tag, _ in top_tags)}" if top_tags else "No tags"
        ]
        
        return ". ".join(summary_parts)
