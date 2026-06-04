import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from clifford.model import Network
from clifford.utils import (
    deserialize_ndarray,
    ensure_directory,
    get_legacy_model_dirs,
    get_models_dir,
    serialize_ndarray,
)
from clifford.core.exceptions import PersistenceError


class Store:
    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = get_models_dir()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_model_dirs = [
            path for path in get_legacy_model_dirs() if path.resolve() != self.models_dir.resolve()
        ]

    def _model_path(self, name: str) -> Path:
        return self.models_dir / f"{name}.pkl"

    def _weight_path(self, name: str) -> Path:
        return self.models_dir / f"{name}_weights.json"

    def _candidate_model_paths(self, name: str) -> List[Path]:
        paths = [self._model_path(name)]
        paths.extend(path / f"{name}.pkl" for path in self.legacy_model_dirs)
        return paths

    def _find_model_path(self, name: str) -> Optional[Path]:
        for path in self._candidate_model_paths(name):
            if path.exists():
                return path
        return None

    def save(self, network: Network, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        model_path = self._model_path(name)
        
        model_data = {
            "network": network,
            "architecture": network.get_architecture(),
            "params": network.get_params(),
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            return str(model_path)
        except Exception as e:
            raise PersistenceError(f"Failed to save model: {e}")

    def load(self, name: str) -> Network:
        model_path = self._find_model_path(name)
        
        if model_path is None:
            candidates = ", ".join(str(path) for path in self._candidate_model_paths(name))
            raise PersistenceError(f"Model not found. Checked: {candidates}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            network = model_data["network"]
            return network
        except Exception as e:
            raise PersistenceError(f"Failed to load model: {e}")

    def save_weights(self, network: Network, name: str) -> str:
        weights_path = self._weight_path(name)
        
        params = network.get_params()
        serialized_params = {}
        
        for key, value in params.items():
            serialized_params[key] = serialize_ndarray(value)
        
        weights_data = {
            "architecture": network.get_architecture(),
            "weights": serialized_params,
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(weights_path, 'w') as f:
                json.dump(weights_data, f, indent=2)
            return str(weights_path)
        except Exception as e:
            raise PersistenceError(f"Failed to save weights: {e}")

    def load_weights(self, network: Network, name: str) -> None:
        weights_path = self._weight_path(name)
        
        if not weights_path.exists():
            raise PersistenceError(f"Weights not found: {weights_path}")
        
        try:
            with open(weights_path, 'r') as f:
                weights_data = json.load(f)
            
            serialized_weights = weights_data["weights"]
            params = {}
            
            for key, value in serialized_weights.items():
                params[key] = deserialize_ndarray(value)
            
            network.set_params(params)
        except Exception as e:
            raise PersistenceError(f"Failed to load weights: {e}")

    def save_checkpoint(self, network: Network, run_id: str, epoch: int, loss: float, metrics: Dict[str, float]) -> str:
        checkpoint_dir = self.models_dir / "checkpoints" / run_id
        ensure_directory(checkpoint_dir)
        
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pkl"
        
        checkpoint_data = {
            "network": network,
            "params": network.get_params(),
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            return str(checkpoint_path)
        except Exception as e:
            raise PersistenceError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, run_id: str, epoch: int) -> Network:
        checkpoint_path = self.models_dir / "checkpoints" / run_id / f"epoch_{epoch}.pkl"
        
        if not checkpoint_path.exists():
            raise PersistenceError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            return checkpoint_data["network"]
        except Exception as e:
            raise PersistenceError(f"Failed to load checkpoint: {e}")

    def list(self) -> List[str]:
        models = set()
        for directory in [self.models_dir, *self.legacy_model_dirs]:
            if not directory.exists():
                continue
            for path in directory.glob("*.pkl"):
                models.add(path.stem)
        return sorted(models)

    def list_paths(self) -> Dict[str, Path]:
        models = {}
        for directory in [*self.legacy_model_dirs, self.models_dir]:
            if not directory.exists():
                continue
            for path in directory.glob("*.pkl"):
                models[path.stem] = path
        return dict(sorted(models.items()))

    def get_metadata(self, name: str) -> Dict[str, Any]:
        model_path = self._find_model_path(name)
        if model_path is None:
            raise PersistenceError(f"Model not found: {name}")

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, dict):
                return {
                    "path": str(model_path),
                    "architecture": model_data.get("architecture"),
                    "metadata": model_data.get("metadata", {}),
                    "saved_at": model_data.get("saved_at"),
                }

            return {
                "path": str(model_path),
                "architecture": model_data.get_architecture() if hasattr(model_data, "get_architecture") else None,
                "metadata": {},
                "saved_at": None,
            }
        except Exception as e:
            raise PersistenceError(f"Failed to read model metadata: {e}")

    def delete(self, name: str) -> None:
        model_path = self.models_dir / f"{name}.pkl"
        weights_path = self.models_dir / f"{name}_weights.json"
        
        if model_path.exists():
            model_path.unlink()
        
        if weights_path.exists():
            weights_path.unlink()

        for directory in self.legacy_model_dirs:
            legacy_model_path = directory / f"{name}.pkl"
            legacy_weights_path = directory / f"{name}_weights.json"
            if legacy_model_path.exists():
                legacy_model_path.unlink()
            if legacy_weights_path.exists():
                legacy_weights_path.unlink()

    def exists(self, name: str) -> bool:
        return self._find_model_path(name) is not None
