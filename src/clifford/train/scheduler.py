from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
import threading
import time

from clifford.model import Network
from clifford.train import Trainer
from clifford.core.types import TrainingConfig


class TrainingScheduler:
    def __init__(self):
        self.scheduled_runs: List[Dict[str, Any]] = []
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []

    def schedule_training(self, network: Network, X, y, config: Optional[TrainingConfig] = None, 
                          start_time: Optional[datetime] = None, repeat_interval: Optional[timedelta] = None) -> str:
        import uuid
        run_id = str(uuid.uuid4())
        
        scheduled_run = {
            "id": run_id,
            "network": network,
            "X": X,
            "y": y,
            "config": config or TrainingConfig(),
            "start_time": start_time or datetime.now(),
            "repeat_interval": repeat_interval,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }
        
        self.scheduled_runs.append(scheduled_run)
        return run_id

    def cancel_scheduled_run(self, run_id: str) -> bool:
        for run in self.scheduled_runs:
            if run["id"] == run_id and run["status"] == "scheduled":
                run["status"] = "cancelled"
                return True
        return False

    def get_scheduled_runs(self) -> List[Dict[str, Any]]:
        return [run for run in self.scheduled_runs if run["status"] == "scheduled"]

    def get_run_status(self, run_id: str) -> Optional[str]:
        for run in self.scheduled_runs:
            if run["id"] == run_id:
                return run["status"]
        return None

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

    def add_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)

    def _run_scheduler(self) -> None:
        while self.running:
            now = datetime.now()
            
            for run in self.scheduled_runs:
                if run["status"] == "scheduled" and now >= run["start_time"]:
                    self._execute_scheduled_run(run)
            
            time.sleep(1)

    def _execute_scheduled_run(self, run: Dict[str, Any]) -> None:
        run["status"] = "running"
        
        try:
            trainer = Trainer(run["network"], run["config"])
            trainer.train(run["X"], run["y"], epochs=run["config"].epochs, verbose=False)
            
            run["status"] = "completed"
            run["completed_at"] = datetime.now().isoformat()
            
            for callback in self.callbacks:
                callback(run)
            
            if run["repeat_interval"]:
                new_start_time = datetime.now() + run["repeat_interval"]
                self.schedule_training(
                    run["network"], run["X"], run["y"], run["config"],
                    new_start_time, run["repeat_interval"]
                )
        except Exception as e:
            run["status"] = "failed"
            run["error"] = str(e)
