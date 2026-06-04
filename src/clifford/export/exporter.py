from typing import Dict, Any, Optional
import json
import pickle
from pathlib import Path
import sqlite3

from clifford.model import Network
from clifford.database import Registry


class Exporter:
    def __init__(self, export_dir: Optional[Path] = None):
        if export_dir is None:
            export_dir = Path(__file__).parent.parent.parent.parent / "exports"
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_model(self, network: Network, name: str, format: str = "json") -> str:
        if format == "json":
            return self._export_model_json(network, name)
        elif format == "pickle":
            return self._export_model_pickle(network, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_model_json(self, network: Network, name: str) -> str:
        export_path = self.export_dir / f"{name}.json"
        
        data = {
            "name": network.name,
            "architecture": network.get_architecture(),
            "params": {k: v.tolist() for k, v in network.get_params().items()},
            "compiled": network.compiled,
            "input_dim": network.input_dim,
        }
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(export_path)

    def _export_model_pickle(self, network: Network, name: str) -> str:
        export_path = self.export_dir / f"{name}.pkl"
        
        with open(export_path, 'wb') as f:
            pickle.dump(network, f)
        
        return str(export_path)

    def export_config(self, config: Dict[str, Any], name: str) -> str:
        export_path = self.export_dir / f"{name}_config.json"
        
        with open(export_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(export_path)

    def export_metrics(self, metrics: Dict[str, Any], name: str) -> str:
        export_path = self.export_dir / f"{name}_metrics.json"
        
        with open(export_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return str(export_path)

    def export_database(self, registry: Registry, name: str) -> str:
        export_path = self.export_dir / f"{name}_db.sqlite"
        
        import shutil
        shutil.copy(registry.db_path, export_path)
        
        return str(export_path)

    def export_training_data(self, X, y, name: str) -> str:
        import numpy as np
        
        export_path = self.export_dir / f"{name}_data.npz"
        
        np.savez(export_path, X=X, y=y)
        
        return str(export_path)
