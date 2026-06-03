import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import json

from clifford.utils import get_db_path, ensure_directory
from clifford.exceptions import DatabaseError
from clifford.types import ModelMetadata, TrainingRun, Checkpoint, DatasetInfo, MetricRecord


class ModelRegistry:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_db_path()
        self.db_path = Path(db_path)
        ensure_directory(self.db_path.parent)
        self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    architecture TEXT NOT NULL,
                    hyperparameters TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    config TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    final_loss REAL,
                    final_metrics TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES training_runs (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    samples INTEGER NOT NULL,
                    features INTEGER NOT NULL,
                    classes INTEGER,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    metrics TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    path TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES training_runs (id)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_model_id ON training_runs(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id)")
            
            conn.commit()

    def register_model(self, name: str, architecture: Dict[str, Any], hyperparameters: Dict[str, Any]) -> str:
        model_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO models (id, name, architecture, hyperparameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (model_id, name, json.dumps(architecture), json.dumps(hyperparameters), now, now)
                )
                conn.commit()
                return model_id
            except sqlite3.IntegrityError:
                raise DatabaseError(f"Model with name '{name}' already exists")

    def get_model(self, name: str) -> Optional[ModelMetadata]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, architecture, hyperparameters, created_at, updated_at FROM models WHERE name = ?", (name,))
            row = cursor.fetchone()
            
            if row:
                return ModelMetadata(
                    id=row[0],
                    name=row[1],
                    architecture=row[2],
                    hyperparameters=json.loads(row[3]),
                    created_at=row[4],
                    updated_at=row[5]
                )
            return None

    def list_models(self) -> List[ModelMetadata]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, architecture, hyperparameters, created_at, updated_at FROM models")
            rows = cursor.fetchall()
            
            return [
                ModelMetadata(
                    id=row[0],
                    name=row[1],
                    architecture=row[2],
                    hyperparameters=json.loads(row[3]),
                    created_at=row[4],
                    updated_at=row[5]
                )
                for row in rows
            ]

    def update_model(self, name: str, architecture: Optional[Dict[str, Any]] = None, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if architecture and hyperparameters:
                cursor.execute(
                    "UPDATE models SET architecture = ?, hyperparameters = ?, updated_at = ? WHERE name = ?",
                    (json.dumps(architecture), json.dumps(hyperparameters), now, name)
                )
            elif architecture:
                cursor.execute(
                    "UPDATE models SET architecture = ?, updated_at = ? WHERE name = ?",
                    (json.dumps(architecture), now, name)
                )
            elif hyperparameters:
                cursor.execute(
                    "UPDATE models SET hyperparameters = ?, updated_at = ? WHERE name = ?",
                    (json.dumps(hyperparameters), now, name)
                )
            else:
                cursor.execute("UPDATE models SET updated_at = ? WHERE name = ?", (now, name))
            
            conn.commit()

    def delete_model(self, name: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM models WHERE name = ?", (name,))
            conn.commit()

    def register_training_run(self, model_id: str, config: Dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO training_runs (id, model_id, config, start_time, status) VALUES (?, ?, ?, ?, ?)",
                (run_id, model_id, json.dumps(config), start_time, "running")
            )
            conn.commit()
            return run_id

    def update_training_run(self, run_id: str, end_time: Optional[str] = None, status: Optional[str] = None, final_loss: Optional[float] = None, final_metrics: Optional[Dict[str, float]] = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if end_time:
                updates.append("end_time = ?")
                params.append(end_time)
            if status:
                updates.append("status = ?")
                params.append(status)
            if final_loss is not None:
                updates.append("final_loss = ?")
                params.append(final_loss)
            if final_metrics:
                updates.append("final_metrics = ?")
                params.append(json.dumps(final_metrics))
            
            if updates:
                params.append(run_id)
                cursor.execute(f"UPDATE training_runs SET {', '.join(updates)} WHERE id = ?", params)
                conn.commit()

    def get_training_run(self, run_id: str) -> Optional[TrainingRun]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, model_id, config, start_time, end_time, status, final_loss, final_metrics FROM training_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            
            if row:
                return TrainingRun(
                    id=row[0],
                    model_id=row[1],
                    config=json.loads(row[2]),
                    start_time=row[3],
                    end_time=row[4],
                    status=row[5],
                    final_loss=row[6],
                    final_metrics=json.loads(row[7]) if row[7] else None
                )
            return None

    def list_training_runs(self, model_id: Optional[str] = None) -> List[TrainingRun]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_id:
                cursor.execute("SELECT id, model_id, config, start_time, end_time, status, final_loss, final_metrics FROM training_runs WHERE model_id = ?", (model_id,))
            else:
                cursor.execute("SELECT id, model_id, config, start_time, end_time, status, final_loss, final_metrics FROM training_runs")
            
            rows = cursor.fetchall()
            
            return [
                TrainingRun(
                    id=row[0],
                    model_id=row[1],
                    config=json.loads(row[2]),
                    start_time=row[3],
                    end_time=row[4],
                    status=row[5],
                    final_loss=row[6],
                    final_metrics=json.loads(row[7]) if row[7] else None
                )
                for row in rows
            ]

    def register_dataset(self, name: str, samples: int, features: int, classes: Optional[int], path: str) -> str:
        dataset_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO datasets (id, name, samples, features, classes, path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (dataset_id, name, samples, features, classes, path, now)
                )
                conn.commit()
                return dataset_id
            except sqlite3.IntegrityError:
                raise DatabaseError(f"Dataset with name '{name}' already exists")

    def list_datasets(self) -> List[DatasetInfo]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, samples, features, classes, path, created_at FROM datasets")
            rows = cursor.fetchall()
            
            return [
                DatasetInfo(
                    id=row[0],
                    name=row[1],
                    samples=row[2],
                    features=row[3],
                    classes=row[4],
                    path=row[5],
                    created_at=row[6]
                )
                for row in rows
            ]

    def register_checkpoint(self, run_id: str, epoch: int, loss: float, metrics: Dict[str, float], path: str) -> str:
        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO checkpoints (id, run_id, epoch, loss, metrics, timestamp, path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (checkpoint_id, run_id, epoch, loss, json.dumps(metrics), timestamp, path)
            )
            conn.commit()
            return checkpoint_id

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, run_id, epoch, loss, metrics, timestamp, path FROM checkpoints WHERE run_id = ? ORDER BY epoch", (run_id,))
            rows = cursor.fetchall()
            
            return [
                Checkpoint(
                    id=row[0],
                    run_id=row[1],
                    epoch=row[2],
                    loss=row[3],
                    metrics=json.loads(row[4]),
                    timestamp=row[5],
                    path=row[6]
                )
                for row in rows
            ]

    def log_metric(self, run_id: str, epoch: int, metric_name: str, value: float) -> None:
        metric_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO metrics (id, run_id, epoch, metric_name, value, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (metric_id, run_id, epoch, metric_name, value, timestamp)
            )
            conn.commit()

    def get_metrics(self, run_id: str) -> List[MetricRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, run_id, epoch, metric_name, value, timestamp FROM metrics WHERE run_id = ? ORDER BY epoch", (run_id,))
            rows = cursor.fetchall()
            
            return [
                MetricRecord(
                    id=row[0],
                    run_id=row[1],
                    epoch=row[2],
                    metric_name=row[3],
                    value=row[4],
                    timestamp=row[5]
                )
                for row in rows
            ]
