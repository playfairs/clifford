from typing import List, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray
import hashlib

from clifford.memory.types import MemoryItem


class MemoryClusterer:
    @staticmethod
    def cluster_by_content(items: List[MemoryItem], n_clusters: int = 5) -> Dict[int, List[MemoryItem]]:
        if not items or len(items) < n_clusters:
            return {0: items}
        
        clusters = {}
        for item in items:
            content_hash = hash(str(item.content)) % n_clusters
            if content_hash not in clusters:
                clusters[content_hash] = []
            clusters[content_hash].append(item)
        
        return clusters

    @staticmethod
    def cluster_by_tags(items: List[MemoryItem]) -> Dict[str, List[MemoryItem]]:
        clusters = {}
        
        for item in items:
            if not item.tags:
                if "untagged" not in clusters:
                    clusters["untagged"] = []
                clusters["untagged"].append(item)
            else:
                for tag in item.tags:
                    if tag not in clusters:
                        clusters[tag] = []
                    clusters[tag].append(item)
        
        return clusters

    @staticmethod
    def cluster_by_importance(items: List[MemoryItem], n_bins: int = 3) -> Dict[str, List[MemoryItem]]:
        if not items:
            return {}
        
        importances = [item.importance for item in items]
        min_imp = min(importances)
        max_imp = max(importances)
        
        if min_imp == max_imp:
            return {"medium": items}
        
        bin_size = (max_imp - min_imp) / n_bins
        clusters = {}
        
        for item in items:
            bin_index = int((item.importance - min_imp) / bin_size)
            bin_index = min(bin_index, n_bins - 1)
            
            if bin_index == 0:
                cluster_name = "low"
            elif bin_index == n_bins - 1:
                cluster_name = "high"
            else:
                cluster_name = "medium"
            
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(item)
        
        return clusters

    @staticmethod
    def cluster_by_time(items: List[MemoryItem], time_window_hours: int = 24) -> Dict[str, List[MemoryItem]]:
        from datetime import datetime, timedelta
        
        clusters = {}
        current_time = datetime.now()
        
        for item in items:
            try:
                item_time = datetime.fromisoformat(item.timestamp)
                time_diff = (current_time - item_time).total_seconds()
                hours_diff = time_diff / 3600
                
                if hours_diff < 1:
                    cluster_name = "last_hour"
                elif hours_diff < 24:
                    cluster_name = "last_day"
                elif hours_diff < 168:
                    cluster_name = "last_week"
                elif hours_diff < 720:
                    cluster_name = "last_month"
                else:
                    cluster_name = "older"
                
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(item)
            except:
                if "unknown" not in clusters:
                    clusters["unknown"] = []
                clusters["unknown"].append(item)
        
        return clusters

    @staticmethod
    def get_cluster_summary(clusters: Dict[Any, List[MemoryItem]]) -> Dict[str, Any]:
        summary = {}
        
        for cluster_id, items in clusters.items():
            summary[str(cluster_id)] = {
                "count": len(items),
                "avg_importance": np.mean([item.importance for item in items]) if items else 0,
                "total_importance": sum(item.importance for item in items),
            }
        
        return summary
