from typing import List, Optional, Dict, Any, Callable
import json
from pathlib import Path
from datetime import datetime

from clifford.reason.types import ReasoningStep, Decision, Plan


class ReasoningEngine:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent.parent / "reasoning" / "history.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.reasoning_history: List[Decision] = []
        self.current_plan: Optional[Plan] = None
        self._load()

    def reason(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[ReasoningStep]:
        steps = []
        
        step1 = ReasoningStep(
            thought=f"Analyzing query: {query}",
            evidence=[f"Context: {context}" if context else "No context provided"],
            confidence=0.8
        )
        steps.append(step1)
        
        step2 = ReasoningStep(
            thought="Generating possible approaches",
            evidence=["Approach 1: Direct analysis", "Approach 2: Contextual analysis"],
            confidence=0.7
        )
        steps.append(step2)
        
        step3 = ReasoningStep(
            thought="Selecting best approach based on available information",
            evidence=["Selected: Contextual analysis" if context else "Selected: Direct analysis"],
            confidence=0.75
        )
        steps.append(step3)
        
        return steps

    def make_decision(self, action: str, reasoning_steps: List[ReasoningStep], alternatives: Optional[List[str]] = None) -> Decision:
        avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.5
        
        decision = Decision(
            action=action,
            reasoning=reasoning_steps,
            confidence=avg_confidence,
            alternatives=alternatives or []
        )
        
        self.reasoning_history.append(decision)
        self._save()
        
        return decision

    def reflect(self, decision: Decision, outcome: str) -> Dict[str, Any]:
        decision.outcome = outcome
        self._save()
        
        reflection = {
            "decision_id": decision.id,
            "action": decision.action,
            "outcome": outcome,
            "confidence": decision.confidence,
            "reasoning_steps": len(decision.reasoning),
            "timestamp": decision.timestamp,
            "reflection": f"Decision '{decision.action}' resulted in '{outcome}' with confidence {decision.confidence:.2f}"
        }
        
        return reflection

    def create_plan(self, goal: str, steps: List[str]) -> Plan:
        plan = Plan(
            goal=goal,
            steps=steps,
            current_step=0,
            status="pending"
        )
        
        self.current_plan = plan
        self._save()
        
        return plan

    def execute_plan_step(self) -> Optional[str]:
        if not self.current_plan:
            return None
        
        if self.current_plan.current_step >= len(self.current_plan.steps):
            self.current_plan.status = "completed"
            self._save()
            return None
        
        step = self.current_plan.steps[self.current_plan.current_step]
        self.current_plan.current_step += 1
        self.current_plan.status = "in_progress"
        self._save()
        
        return step

    def complete_plan(self) -> None:
        if self.current_plan:
            self.current_plan.status = "completed"
            self._save()

    def evaluate_decision(self, decision: Decision, evaluation_func: Callable[[Decision], float]) -> float:
        return evaluation_func(decision)

    def compare_decisions(self, decisions: List[Decision]) -> Dict[str, Any]:
        if not decisions:
            return {}
        
        return {
            "count": len(decisions),
            "avg_confidence": sum(d.confidence for d in decisions) / len(decisions),
            "highest_confidence": max(d.confidence for d in decisions),
            "lowest_confidence": min(d.confidence for d in decisions),
            "decisions": [{"id": d.id, "action": d.action, "confidence": d.confidence} for d in decisions]
        }

    def get_reasoning_history(self, limit: int = 10) -> List[Decision]:
        return self.reasoning_history[-limit:]

    def clear_history(self) -> None:
        self.reasoning_history.clear()
        self._save()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for decision_data in data:
                        reasoning_steps = [ReasoningStep(**step) for step in decision_data.get("reasoning", [])]
                        decision = Decision(
                            id=decision_data.get("id"),
                            action=decision_data.get("action"),
                            reasoning=reasoning_steps,
                            confidence=decision_data.get("confidence", 0.5),
                            alternatives=decision_data.get("alternatives", []),
                            timestamp=decision_data.get("timestamp"),
                            outcome=decision_data.get("outcome")
                        )
                        self.reasoning_history.append(decision)
            except Exception:
                pass

    def _save(self) -> None:
        data = []
        for decision in self.reasoning_history:
            data.append({
                "id": decision.id,
                "action": decision.action,
                "reasoning": [{"id": step.id, "thought": step.thought, "evidence": step.evidence, "confidence": step.confidence, "timestamp": step.timestamp} for step in decision.reasoning],
                "confidence": decision.confidence,
                "alternatives": decision.alternatives,
                "timestamp": decision.timestamp,
                "outcome": decision.outcome
            })
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
