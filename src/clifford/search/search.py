import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from clifford.utils import get_db_path
from clifford.database import Registry
from clifford.core.exceptions import SearchError
from clifford.core.types import ModelMetadata, TrainingRun, DatasetInfo, Checkpoint


class Search:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_db_path()
        self.db_path = Path(db_path)
        self.registry = Registry(db_path)

    def search_models(self, name: Optional[str] = None, architecture: Optional[str] = None, hyperparameter_filter: Optional[Dict[str, Any]] = None) -> List[ModelMetadata]:
        models = self.registry.list_models()
        
        if name:
            models = [m for m in models if name.lower() in m.name.lower()]
        
        if architecture:
            models = [m for m in models if architecture.lower() in m.architecture.lower()]
        
        if hyperparameter_filter:
            filtered_models = []
            for model in models:
                match = True
                for key, value in hyperparameter_filter.items():
                    if key not in model.hyperparameters or model.hyperparameters[key] != value:
                        match = False
                        break
                if match:
                    filtered_models.append(model)
            models = filtered_models
        
        return models

    def search_training_runs(self, model_id: Optional[str] = None, status: Optional[str] = None, min_loss: Optional[float] = None, max_loss: Optional[float] = None) -> List[TrainingRun]:
        runs = self.registry.list_training_runs(model_id)
        
        if status:
            runs = [r for r in runs if r.status == status]
        
        if min_loss is not None:
            runs = [r for r in runs if r.final_loss is not None and r.final_loss >= min_loss]
        
        if max_loss is not None:
            runs = [r for r in runs if r.final_loss is not None and r.final_loss <= max_loss]
        
        return runs

    def search_datasets(self, name: Optional[str] = None, min_samples: Optional[int] = None, max_samples: Optional[int] = None, features: Optional[int] = None) -> List[DatasetInfo]:
        datasets = self.registry.list_datasets()
        
        if name:
            datasets = [d for d in datasets if name.lower() in d.name.lower()]
        
        if min_samples is not None:
            datasets = [d for d in datasets if d.samples >= min_samples]
        
        if max_samples is not None:
            datasets = [d for d in datasets if d.samples <= max_samples]
        
        if features is not None:
            datasets = [d for d in datasets if d.features == features]
        
        return datasets

    def search_checkpoints(self, run_id: str, min_epoch: Optional[int] = None, max_epoch: Optional[int] = None, min_loss: Optional[float] = None, max_loss: Optional[float] = None) -> List[Checkpoint]:
        checkpoints = self.registry.list_checkpoints(run_id)
        
        if min_epoch is not None:
            checkpoints = [c for c in checkpoints if c.epoch >= min_epoch]
        
        if max_epoch is not None:
            checkpoints = [c for c in checkpoints if c.epoch <= max_epoch]
        
        if min_loss is not None:
            checkpoints = [c for c in checkpoints if c.loss >= min_loss]
        
        if max_loss is not None:
            checkpoints = [c for c in checkpoints if c.loss <= max_loss]
        
        return checkpoints

    def search_by_date(self, table: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if table == "models":
                date_column = "created_at"
                query = f"SELECT id, name, architecture, hyperparameters, created_at, updated_at FROM models"
            elif table == "training_runs":
                date_column = "start_time"
                query = f"SELECT id, model_id, config, start_time, end_time, status, final_loss, final_metrics FROM training_runs"
            elif table == "datasets":
                date_column = "created_at"
                query = f"SELECT id, name, samples, features, classes, path, created_at FROM datasets"
            elif table == "checkpoints":
                date_column = "timestamp"
                query = f"SELECT id, run_id, epoch, loss, metrics, timestamp, path FROM checkpoints"
            else:
                raise SearchError(f"Unknown table: {table}")
            
            conditions = []
            params = []
            
            if start_date:
                conditions.append(f"{date_column} >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append(f"{date_column} <= ?")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if table == "models":
                return [
                    ModelMetadata(
                        id=row[0],
                        name=row[1],
                        architecture=row[2],
                        hyperparameters=eval(row[3]),
                        created_at=row[4],
                        updated_at=row[5]
                    )
                    for row in rows
                ]
            elif table == "training_runs":
                return [
                    TrainingRun(
                        id=row[0],
                        model_id=row[1],
                        config=eval(row[2]),
                        start_time=row[3],
                        end_time=row[4],
                        status=row[5],
                        final_loss=row[6],
                        final_metrics=eval(row[7]) if row[7] else None
                    )
                    for row in rows
                ]
            elif table == "datasets":
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
            elif table == "checkpoints":
                return [
                    Checkpoint(
                        id=row[0],
                        run_id=row[1],
                        epoch=row[2],
                        loss=row[3],
                        metrics=eval(row[4]),
                        timestamp=row[5],
                        path=row[6]
                    )
                    for row in rows
                ]
            
            return []

    def full_text_search(self, query: str) -> Dict[str, List[Any]]:
        results = {
            "models": [],
            "training_runs": [],
            "datasets": [],
            "checkpoints": []
        }
        
        query_lower = query.lower()
        
        models = self.registry.list_models()
        for model in models:
            if query_lower in model.name.lower() or query_lower in model.architecture.lower():
                results["models"].append(model)
        
        runs = self.registry.list_training_runs()
        for run in runs:
            if query_lower in run.status.lower() or query_lower in run.id.lower():
                results["training_runs"].append(run)
        
        datasets = self.registry.list_datasets()
        for dataset in datasets:
            if query_lower in dataset.name.lower():
                results["datasets"].append(dataset)
        
        return results
