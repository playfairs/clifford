from typing import Any, Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime

from clifford.memory.types import EpisodicMemory


class EpisodicMemoryStore:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent.parent / "memory" / "episodic.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.episodes: Dict[str, EpisodicMemory] = {}
        self._load()

    def add_episode(self, episode: str, context: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> EpisodicMemory:
        episodic_memory = EpisodicMemory(
            episode=episode,
            context=context or {},
            importance=importance
        )
        self.episodes[episodic_memory.id] = episodic_memory
        self._save()
        return episodic_memory

    def get_episode(self, episode_id: str) -> Optional[EpisodicMemory]:
        return self.episodes.get(episode_id)

    def search_episodes(self, query: str) -> List[EpisodicMemory]:
        query_lower = query.lower()
        results = []
        
        for episode in self.episodes.values():
            episode_str = episode.episode.lower()
            context_str = str(episode.context).lower()
            
            if query_lower in episode_str or query_lower in context_str:
                results.append(episode)
        
        return sorted(results, key=lambda x: x.importance, reverse=True)

    def get_recent_episodes(self, count: int = 10) -> List[EpisodicMemory]:
        sorted_episodes = sorted(self.episodes.values(), key=lambda x: x.timestamp, reverse=True)
        return sorted_episodes[:count]

    def get_by_importance(self, min_importance: float = 0.0) -> List[EpisodicMemory]:
        return [ep for ep in self.episodes.values() if ep.importance >= min_importance]

    def remove_episode(self, episode_id: str) -> None:
        if episode_id in self.episodes:
            del self.episodes[episode_id]
            self._save()

    def clear(self) -> None:
        self.episodes.clear()
        self._save()

    def size(self) -> int:
        return len(self.episodes)

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for episode_data in data:
                        episode = EpisodicMemory(**episode_data)
                        self.episodes[episode.id] = episode
            except Exception:
                pass

    def _save(self) -> None:
        data = []
        for episode in self.episodes.values():
            data.append({
                "id": episode.id,
                "episode": episode.episode,
                "context": episode.context,
                "timestamp": episode.timestamp,
                "importance": episode.importance,
            })
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
