import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from clifford.network import Network
from clifford.utils import serialize_ndarray, deserialize_ndarray, ensure_directory
from clifford.exceptions import PersistenceError


class ModelPersistence:
    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir = Path(models_dir)
        ensure_directory(self.models_dir)

    def save_model(self, network: Network, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        model_path = self.models_dir / f"{name}.pkl"
        
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

    def load_model(self, name: str) -> Network:
        model_path = self.models_dir / f"{name}.pkl"
        
        if not model_path.exists():
            raise PersistenceError(f"Model not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            network = model_data["network"]
            return network
        except Exception as e:
            raise PersistenceError(f"Failed to load model: {e}")

    def save_weights(self, network: Network, name: str) -> str:
        weights_path = self.models_dir / f"{name}_weights.json"
        
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
        weights_path = self.models_dir / f"{name}_weights.json"
        
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

    def list_models(self) -> List[str]:
        models = []
        for path in self.models_dir.glob("*.pkl"):
            if not path.name.endswith("_weights.json"):
                models.append(path.stem)
        return models

    def delete_model(self, name: str) -> None:
        model_path = self.models_dir / f"{name}.pkl"
        weights_path = self.models_dir / f"{name}_weights.json"
        
        if model_path.exists():
            model_path.unlink()
        
        if weights_path.exists():
            weights_path.unlink()

    def model_exists(self, name: str) -> bool:
        model_path = self.models_dir / f"{name}.pkl"
        return model_path.exists()
