from typing import List, Optional, Dict, Any, Set
import json
from pathlib import Path
from collections import defaultdict

from clifford.graph.types import Entity, Relationship, Fact


class KnowledgeGraph:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent.parent / "knowledge" / "graph.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.facts: Dict[str, Fact] = {}
        self._load()

    def add_entity(self, name: str, entity_type: str, attributes: Optional[Dict[str, Any]] = None, confidence: float = 1.0) -> Entity:
        entity = Entity(
            name=name,
            entity_type=entity_type,
            attributes=attributes or {},
            confidence=confidence
        )
        self.entities[entity.id] = entity
        self._save()
        return entity

    def add_relationship(self, source: str, target: str, relation_type: str, attributes: Optional[Dict[str, Any]] = None, confidence: float = 1.0) -> Relationship:
        relationship = Relationship(
            source=source,
            target=target,
            relation_type=relation_type,
            attributes=attributes or {},
            confidence=confidence
        )
        self.relationships[relationship.id] = relationship
        self._save()
        return relationship

    def add_fact(self, statement: str, entities: List[str], confidence: float = 1.0, source: str = "") -> Fact:
        fact = Fact(
            statement=statement,
            entities=entities,
            confidence=confidence,
            source=source
        )
        self.facts[fact.id] = fact
        self._save()
        return fact

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        for entity in self.entities.values():
            if entity.name == name:
                return entity
        return None

    def get_relationships(self, entity_id: str) -> List[Relationship]:
        return [r for r in self.relationships.values() if r.source == entity_id or r.target == entity_id]

    def get_related_entities(self, entity_id: str) -> List[Entity]:
        related_ids = set()
        for rel in self.relationships.values():
            if rel.source == entity_id:
                related_ids.add(rel.target)
            elif rel.target == entity_id:
                related_ids.add(rel.source)
        
        return [self.entities[eid] for eid in related_ids if eid in self.entities]

    def search_entities(self, query: str) -> List[Entity]:
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            if query_lower in entity.name.lower() or query_lower in entity.entity_type.lower():
                results.append(entity)
        
        return results

    def search_facts(self, query: str) -> List[Fact]:
        query_lower = query.lower()
        results = []
        
        for fact in self.facts.values():
            if query_lower in fact.statement.lower():
                results.append(fact)
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def get_entity_neighbors(self, entity_id: str, depth: int = 1) -> Dict[str, Set[str]]:
        neighbors = {str(depth): set()}
        current_level = {entity_id}
        
        for d in range(depth):
            next_level = set()
            for eid in current_level:
                for rel in self.relationships.values():
                    if rel.source == eid:
                        next_level.add(rel.target)
                    elif rel.target == eid:
                        next_level.add(rel.source)
            neighbors[str(d + 1)] = next_level
            current_level = next_level
        
        return neighbors

    def get_path(self, source_id: str, target_id: str) -> List[str]:
        from collections import deque
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_id:
                return path
            
            for rel in self.relationships.values():
                if rel.source == current and rel.target not in visited:
                    visited.add(rel.target)
                    queue.append((rel.target, path + [rel.target]))
                elif rel.target == current and rel.source not in visited:
                    visited.add(rel.source)
                    queue.append((rel.source, path + [rel.source]))
        
        return []

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.entities:
            del self.entities[entity_id]
            self.relationships = {k: v for k, v in self.relationships.items() if v.source != entity_id and v.target != entity_id}
            self._save()

    def remove_relationship(self, relationship_id: str) -> None:
        if relationship_id in self.relationships:
            del self.relationships[relationship_id]
            self._save()

    def remove_fact(self, fact_id: str) -> None:
        if fact_id in self.facts:
            del self.facts[fact_id]
            self._save()

    def clear(self) -> None:
        self.entities.clear()
        self.relationships.clear()
        self.facts.clear()
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "fact_count": len(self.facts),
            "entity_types": len(set(e.entity_type for e in self.entities.values())),
            "relationship_types": len(set(r.relation_type for r in self.relationships.values())),
        }

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                    for entity_data in data.get("entities", []):
                        entity = Entity(**entity_data)
                        self.entities[entity.id] = entity
                    
                    for rel_data in data.get("relationships", []):
                        rel = Relationship(**rel_data)
                        self.relationships[rel.id] = rel
                    
                    for fact_data in data.get("facts", []):
                        fact = Fact(**fact_data)
                        self.facts[fact.id] = fact
            except Exception:
                pass

    def _save(self) -> None:
        data = {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "attributes": e.attributes,
                    "confidence": e.confidence,
                    "timestamp": e.timestamp
                }
                for e in self.entities.values()
            ],
            "relationships": [
                {
                    "id": r.id,
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "attributes": r.attributes,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp
                }
                for r in self.relationships.values()
            ],
            "facts": [
                {
                    "id": f.id,
                    "statement": f.statement,
                    "entities": f.entities,
                    "confidence": f.confidence,
                    "source": f.source,
                    "timestamp": f.timestamp
                }
                for f in self.facts.values()
            ]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
