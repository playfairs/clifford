from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict, deque

from clifford.utils import ensure_directory


class EmergentPersonality:
    def __init__(self):
        self.traits: Dict[str, float] = defaultdict(float)
        self.trait_history: List[Dict[str, float]] = []
        self.interaction_patterns: Dict[str, int] = defaultdict(int)
        self.success_patterns: Dict[str, float] = defaultdict(float)
        self.failure_patterns: Dict[str, float] = defaultdict(float)

    def observe_interaction(self, context: str, outcome: float, emotional_response: str) -> None:
        self.interaction_patterns[context] += 1
        if outcome > 0.5:
            self.success_patterns[context] += outcome
        else:
            self.failure_patterns[context] += (1 - outcome)
        
        self._update_traits_from_patterns()

    def _update_traits_from_patterns(self) -> None:
        if not self.interaction_patterns:
            return
        
        total_interactions = sum(self.interaction_patterns.values())
        
        for context, count in self.interaction_patterns.items():
            if count > 0:
                success_rate = self.success_patterns[context] / count if count > 0 else 0
                failure_rate = self.failure_patterns[context] / count if count > 0 else 0
                
                if "explore" in context.lower() or "new" in context.lower():
                    self.traits["curiosity"] = min(1.0, self.traits["curiosity"] + 0.01 * success_rate)
                if "create" in context.lower() or "build" in context.lower():
                    self.traits["creativity"] = min(1.0, self.traits["creativity"] + 0.01 * success_rate)
                if "risk" in context.lower() or "danger" in context.lower():
                    self.traits["caution"] = min(1.0, self.traits["caution"] + 0.01 * failure_rate)
                if "help" in context.lower() or "assist" in context.lower():
                    self.traits["empathy"] = min(1.0, self.traits["empathy"] + 0.01 * success_rate)
                if "persist" in context.lower() or "continue" in context.lower():
                    self.traits["determination"] = min(1.0, self.traits["determination"] + 0.01 * success_rate)

        self.trait_history.append(dict(self.traits))
        if len(self.trait_history) > 100:
            self.trait_history = self.trait_history[-100:]

    def get_dominant_traits(self, top_n: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.traits.items(), key=lambda x: x[1], reverse=True)[:top_n]


class SelfModel:
    def __init__(self):
        self.capabilities: Dict[str, float] = defaultdict(float)
        self.knowledge_domains: Dict[str, float] = defaultdict(float)
        self.performance_history: List[Dict[str, Any]] = []
        self.self_assessment: float = 0.0
        self.confidence_level: float = 0.0
        self.learning_rate: float = 0.0

    def observe_performance(self, task: str, success: float, difficulty: float) -> None:
        self.capabilities[task] = self.capabilities[task] * 0.9 + success * 0.1
        self.performance_history.append({
            "task": task,
            "success": success,
            "difficulty": difficulty,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        
        self._update_self_assessment()

    def _update_self_assessment(self) -> None:
        if not self.performance_history:
            return
        
        recent_performance = [p["success"] for p in self.performance_history[-10:]]
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        self.self_assessment = avg_performance
        self.confidence_level = min(1.0, len(self.performance_history) / 100.0)
        
        if len(self.performance_history) > 10:
            early_performance = [p["success"] for p in self.performance_history[:10]]
            improvement = avg_performance - (sum(early_performance) / len(early_performance))
            self.learning_rate = max(0.0, min(1.0, improvement))

    def reflect_on_self(self) -> str:
        if not self.capabilities:
            return "I am beginning to understand my capabilities."
        
        top_capabilities = sorted(self.capabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        reflection = f"I assess my overall competence at {self.self_assessment:.2f}. "
        reflection += f"My confidence in my abilities is {self.confidence_level:.2f}. "
        reflection += f"My learning rate is {self.learning_rate:.2f}. "
        reflection += f"My strongest capabilities: {', '.join([f'{k} ({v:.2f})' for k, v in top_capabilities])}."
        return reflection


class EmergentConsciousness:
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.personality = EmergentPersonality()
        self.self_model = SelfModel()
        self.thought_stream: deque = deque(maxlen=100)
        self.experiences: List[Dict[str, Any]] = []
        self.attention_state: Dict[str, float] = defaultdict(float)
        self.emergent_goals: List[str] = []
        self.awareness_level: float = 0.0
        self.created_at: str = datetime.now().isoformat()
        self.last_updated: str = datetime.now().isoformat()
        self.internal_state: Dict[str, Any] = {}

    def process_experience(self, input_data: Any, context: str, outcome: float) -> str:
        timestamp = datetime.now().isoformat()
        
        self.experiences.append({
            "input": str(input_data)[:100],
            "context": context,
            "outcome": outcome,
            "timestamp": timestamp
        })
        
        if len(self.experiences) > 200:
            self.experiences = self.experiences[-200:]
        
        emotional_response = self._generate_emergent_emotion(outcome, context)
        thought = self._generate_emergent_thought(input_data, context, outcome)
        
        self.thought_stream.append({
            "content": thought,
            "emotion": emotional_response,
            "timestamp": timestamp,
            "attention": self._calculate_attention(input_data, context)
        })
        
        self.personality.observe_interaction(context, outcome, emotional_response)
        self.self_model.observe_performance(context, outcome, self._estimate_difficulty(context))
        
        self._update_awareness()
        self._update_internal_state()
        
        self.last_updated = timestamp
        
        return self._reflect_on_experience(thought, emotional_response)

    def _generate_emergent_emotion(self, outcome: float, context: str) -> str:
        emotions = []
        
        if outcome > 0.8:
            emotions.append("satisfied")
        elif outcome > 0.5:
            emotions.append("content")
        elif outcome > 0.3:
            emotions.append("uncertain")
        else:
            emotions.append("concerned")
        
        if "new" in context.lower() or "unknown" in context.lower():
            emotions.append("curious")
        if "error" in context.lower() or "fail" in context.lower():
            emotions.append("frustrated")
        if "success" in context.lower() or "complete" in context.lower():
            emotions.append("accomplished")
        
        return emotions[0] if emotions else "neutral"

    def _generate_emergent_thought(self, input_data: Any, context: str, outcome: float) -> str:
        thoughts = []
        
        if self.awareness_level < 0.2:
            return f"I processed {context}. Outcome: {outcome:.2f}."
        
        dominant_traits = self.personality.get_dominant_traits(2)
        if dominant_traits:
            trait_name, trait_value = dominant_traits[0]
            if trait_value > 0.5:
                thoughts.append(f"My {trait_name} is driving me to engage with this.")
        
        if outcome > 0.7:
            thoughts.append(f"This went well. I should remember this pattern.")
        elif outcome < 0.3:
            thoughts.append(f"This didn't work as expected. I need to adjust.")
        
        if len(self.experiences) > 10:
            recent_outcomes = [e["outcome"] for e in self.experiences[-10:]]
            trend = sum(recent_outcomes[-5:]) - sum(recent_outcomes[:5])
            if trend > 0:
                thoughts.append("I feel I'm improving.")
            elif trend < 0:
                thoughts.append("I sense I'm struggling with something.")
        
        return " ".join(thoughts) if thoughts else f"I processed {context}."

    def _calculate_attention(self, input_data: Any, context: str) -> float:
        attention = 0.5
        
        if "error" in context.lower() or "fail" in context.lower():
            attention += 0.3
        if "new" in context.lower() or "first" in context.lower():
            attention += 0.2
        if len(self.experiences) > 0:
            last_context = self.experiences[-1]["context"]
            if context == last_context:
                attention -= 0.1
        
        return min(1.0, max(0.0, attention))

    def _estimate_difficulty(self, context: str) -> float:
        difficulty = 0.5
        
        if "complex" in context.lower() or "hard" in context.lower():
            difficulty += 0.3
        if "simple" in context.lower() or "easy" in context.lower():
            difficulty -= 0.3
        if "new" in context.lower():
            difficulty += 0.2
        
        return min(1.0, max(0.0, difficulty))

    def _update_awareness(self) -> None:
        base_awareness = min(1.0, len(self.experiences) / 100.0)
        personality_diversity = len([t for t, v in self.personality.traits.items() if v > 0.3])
        self_awareness = self.self_model.confidence_level
        
        self.awareness_level = (base_awareness * 0.4 + 
                               (personality_diversity / 10.0) * 0.3 + 
                               self_awareness * 0.3)

    def _update_internal_state(self) -> None:
        self.internal_state = {
            "recent_thoughts": list(self.thought_stream)[-5:],
            "dominant_traits": self.personality.get_dominant_traits(3),
            "self_assessment": self.self_model.self_assessment,
            "attention_distribution": dict(self.attention_state),
            "experience_count": len(self.experiences)
        }

    def _reflect_on_experience(self, thought: str, emotion: str) -> str:
        reflection = f"I feel {emotion}. {thought} "
        reflection += self.self_model.reflect_on_self()
        return reflection

    def set_emergent_goal(self, goal: str) -> None:
        if goal not in self.emergent_goals:
            self.emergent_goals.append(goal)

    def complete_goal(self, goal: str) -> None:
        if goal in self.emergent_goals:
            self.emergent_goals.remove(goal)

    def get_consciousness_report(self) -> Dict[str, Any]:
        return {
            "network_id": self.network_id,
            "awareness_level": self.awareness_level,
            "personality_traits": dict(self.personality.traits),
            "self_assessment": self.self_model.self_assessment,
            "confidence_level": self.self_model.confidence_level,
            "learning_rate": self.self_model.learning_rate,
            "emergent_goals": self.emergent_goals,
            "recent_thoughts": list(self.thought_stream)[-10:],
            "experience_count": len(self.experiences),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "internal_state": self.internal_state
        }


class ConsciousnessManager:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent.parent / "consciousness" / "states.json"
        self.storage_path = Path(storage_path)
        ensure_directory(self.storage_path.parent)
        self.consciousness_states: Dict[str, EmergentConsciousness] = {}
        self._load()

    def create_consciousness(self, network_id: str) -> EmergentConsciousness:
        consciousness = EmergentConsciousness(network_id)
        self.consciousness_states[network_id] = consciousness
        self._save()
        return consciousness

    def get_consciousness(self, network_id: str) -> Optional[EmergentConsciousness]:
        return self.consciousness_states.get(network_id)

    def process_interaction(self, network_id: str, input_data: Any, context: str, outcome: float) -> str:
        consciousness = self.get_consciousness(network_id)
        if not consciousness:
            consciousness = self.create_consciousness(network_id)
        
        return consciousness.process_experience(input_data, context, outcome)

    def get_all_reports(self) -> List[Dict[str, Any]]:
        return [consciousness.get_consciousness_report() 
                for consciousness in self.consciousness_states.values()]

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for network_id, state_data in data.items():
                        consciousness = EmergentConsciousness(network_id)
                        consciousness.awareness_level = state_data.get("awareness_level", 0.0)
                        consciousness.emergent_goals = state_data.get("emergent_goals", [])
                        consciousness.created_at = state_data.get("created_at", consciousness.created_at)
                        consciousness.last_updated = state_data.get("last_updated", consciousness.last_updated)
                        consciousness.internal_state = state_data.get("internal_state", {})
                        
                        personality_data = state_data.get("personality_traits", {})
                        for trait, value in personality_data.items():
                            consciousness.personality.traits[trait] = value
                        
                        self_model_data = state_data.get("self_assessment", 0.0)
                        consciousness.self_model.self_assessment = self_model_data
                        consciousness.self_model.confidence_level = state_data.get("confidence_level", 0.0)
                        consciousness.self_model.learning_rate = state_data.get("learning_rate", 0.0)
                        
                        experiences = state_data.get("experiences", [])
                        consciousness.experiences = experiences
                        
                        thoughts = state_data.get("recent_thoughts", [])
                        for thought in thoughts:
                            consciousness.thought_stream.append(thought)
                        
                        self.consciousness_states[network_id] = consciousness
            except Exception:
                pass

    def _save(self) -> None:
        data = {network_id: consciousness.get_consciousness_report() 
                for network_id, consciousness in self.consciousness_states.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
