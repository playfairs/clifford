from typing import Optional, Dict, Any, List
from datetime import datetime
import json
from pathlib import Path

from clifford.data import Dataset


class DatasetRegistry:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent.parent / "datasets" / "registry.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self._load()

    def register_dataset(self, dataset: Dataset, metadata: Optional[Dict[str, Any]] = None) -> str:
        dataset_info = {
            "name": dataset.name,
            "samples": dataset.samples,
            "features": dataset.features,
            "classes": dataset.classes,
            "path": str(dataset.path) if hasattr(dataset, 'path') else "",
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        self.datasets[dataset.name] = dataset_info
        self._save()
        
        return dataset.name

    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        return self.datasets.get(name)

    def list_datasets(self) -> List[Dict[str, Any]]:
        return list(self.datasets.values())

    def update_dataset_version(self, name: str, version: str) -> bool:
        if name in self.datasets:
            self.datasets[name]["version"] = version
            self.datasets[name]["updated_at"] = datetime.now().isoformat()
            self._save()
            return True
        return False

    def delete_dataset(self, name: str) -> bool:
        if name in self.datasets:
            del self.datasets[name]
            self._save()
            return True
        return False

    def search_datasets(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        results = []
        
        for dataset_info in self.datasets.values():
            if query_lower in dataset_info["name"].lower():
                results.append(dataset_info)
        
        return results

    def get_dataset_statistics(self, name: str) -> Optional[Dict[str, Any]]:
        if name in self.datasets:
            dataset_info = self.datasets[name]
            return {
                "name": dataset_info["name"],
                "samples": dataset_info["samples"],
                "features": dataset_info["features"],
                "classes": dataset_info["classes"],
                "version": dataset_info["version"],
                "registered_at": dataset_info["registered_at"]
            }
        return None

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    self.datasets = json.load(f)
            except Exception:
                pass

    def _save(self) -> None:
        with open(self.storage_path, 'w') as f:
            json.dump(self.datasets, f, indent=2)
