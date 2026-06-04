from typing import Dict, Any, Optional
import json
import pickle
from pathlib import Path
import numpy as np

from clifford.model import Network
from clifford.utils import ensure_directory


class Importer:
    def __init__(self, import_dir: Optional[Path] = None):
        if import_dir is None:
            import_dir = Path(__file__).parent.parent.parent.parent / "exports"
        self.import_dir = Path(import_dir)

    def import_model(self, path: str, format: str = "json") -> Optional[Network]:
        if format == "json":
            return self._import_model_json(path)
        elif format == "pickle":
            return self._import_model_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _import_model_json(self, path: str) -> Optional[Network]:
        import_path = Path(path)
        
        if not import_path.exists():
            return None
        
        with open(import_path, 'r') as f:
            data = json.load(f)
        
        network = Network(name=data.get("name", "imported_network"))
        
        for key, value in data.get("params", {}).items():
            network.set_params({key: np.array(value)})
        
        network.compiled = data.get("compiled", False)
        network.input_dim = data.get("input_dim")
        
        return network

    def _import_model_pickle(self, path: str) -> Optional[Network]:
        import_path = Path(path)
        
        if not import_path.exists():
            return None
        
        with open(import_path, 'rb') as f:
            network = pickle.load(f)
        
        return network

    def import_config(self, path: str) -> Optional[Dict[str, Any]]:
        import_path = Path(path)
        
        if not import_path.exists():
            return None
        
        with open(import_path, 'r') as f:
            config = json.load(f)
        
        return config

    def import_metrics(self, path: str) -> Optional[Dict[str, Any]]:
        import_path = Path(path)
        
        if not import_path.exists():
            return None
        
        with open(import_path, 'r') as f:
            metrics = json.load(f)
        
        return metrics

    def import_training_data(self, path: str) -> Optional[tuple]:
        import_path = Path(path)
        
        if not import_path.exists():
            return None
        
        data = np.load(import_path)
        return data['X'], data['y']
