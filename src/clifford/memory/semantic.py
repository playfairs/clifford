from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from clifford.memory.types import SemanticMemory


class SemanticMemoryStore:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent.parent / "memory" / "semantic.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.concepts: Dict[str, SemanticMemory] = {}
        self._load()

    def add_concept(self, concept: str, attributes: Optional[Dict[str, Any]] = None, confidence: float = 1.0) -> SemanticMemory:
        semantic_memory = SemanticMemory(
            concept=concept,
            attributes=attributes or {},
            confidence=confidence
        )
        self.concepts[semantic_memory.id] = semantic_memory
        self._save()
        return semantic_memory

    def get_concept(self, concept_id: str) -> Optional[SemanticMemory]:
        return self.concepts.get(concept_id)

    def search_concepts(self, query: str) -> List[SemanticMemory]:
        query_lower = query.lower()
        results = []
        
        for concept in self.concepts.values():
            concept_str = concept.concept.lower()
            attributes_str = str(concept.attributes).lower()
            
            if query_lower in concept_str or query_lower in attributes_str:
                results.append(concept)
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def get_related_concepts(self, concept_id: str) -> List[SemanticMemory]:
        concept = self.concepts.get(concept_id)
        if not concept:
            return []
        
        related = []
        for rel_id in concept.relationships:
            if rel_id in self.concepts:
                related.append(self.concepts[rel_id])
        
        return related

    def add_relationship(self, concept_id: str, related_id: str) -> None:
        if concept_id in self.concepts and related_id in self.concepts:
            if related_id not in self.concepts[concept_id].relationships:
                self.concepts[concept_id].relationships.append(related_id)
            if concept_id not in self.concepts[related_id].relationships:
                self.concepts[related_id].relationships.append(concept_id)
            self._save()

    def remove_concept(self, concept_id: str) -> None:
        if concept_id in self.concepts:
            del self.concepts[concept_id]
            self._save()

    def clear(self) -> None:
        self.concepts.clear()
        self._save()

    def size(self) -> int:
        return len(self.concepts)

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for concept_data in data:
                        concept = SemanticMemory(**concept_data)
                        self.concepts[concept.id] = concept
            except Exception:
                pass

    def _save(self) -> None:
        data = []
        for concept in self.concepts.values():
            data.append({
                "id": concept.id,
                "concept": concept.concept,
                "attributes": concept.attributes,
                "relationships": concept.relationships,
                "confidence": concept.confidence,
            })
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
