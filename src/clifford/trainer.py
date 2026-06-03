import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable, Dict, List, Any
from datetime import datetime
import uuid

from clifford.network import Network
from clifford.types import TrainingConfig, BatchMode
from clifford.utils import create_batches, shuffle_data, split_data
from clifford.exceptions import TrainingError


class Trainer:
    def __init__(self, network: Network, config: Optional[TrainingConfig] = None):
        self.network = network
        self.config = config if config else TrainingConfig()
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        self.callbacks: List[Callable] = []
        self.early_stopping_counter: int = 0
        self.best_loss: float = float('inf')

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, Any]:
        if not self.network.compiled:
            raise TrainingError("Network must be compiled before training")
        
        if self.config.validation_split > 0.0:
            X_train, X_val, y_train, y_val = split_data(X, y, self.config.validation_split)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        run_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()
        
        for epoch in range(self.config.epochs):
            if self.config.shuffle:
                X_train, y_train = shuffle_data(X_train, y_train)
            
            epoch_loss = self._train_epoch(X_train, y_train)
            self.history["loss"].append(epoch_loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                
                if self.config.early_stopping_patience > 0:
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.config.early_stopping_patience:
                            if self.config.verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break
            
            if self.config.verbose:
                val_str = f", val_loss: {val_loss:.6f}" if X_val is not None else ""
                print(f"Epoch {epoch + 1}/{self.config.epochs}, loss: {epoch_loss:.6f}{val_str}")
            
            for callback in self.callbacks:
                callback(epoch, self.history)

        end_time = datetime.now().isoformat()
        
        return {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "history": self.history,
            "final_loss": self.history["loss"][-1],
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None
        }

    def _train_epoch(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        batch_size = self.config.batch_size
        
        if batch_size >= len(X):
            batches = [(X, y)]
        else:
            batches = create_batches(X, y, batch_size)
        
        epoch_loss = 0.0
        for batch_X, batch_y in batches:
            loss = self._train_batch(batch_X, batch_y)
            epoch_loss += loss
        
        return epoch_loss / len(batches)

    def _train_batch(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        y_pred = self.network.forward(X, training=True)
        
        if self.network.loss is None:
            raise TrainingError("Network not compiled with loss function")
        
        loss_value = np.mean(self.network.loss.forward(y_pred, y))
        grad = self.network.loss.backward(y_pred, y)
        
        self.network.backward(grad)
        
        params = self.network.get_params()
        grads = self.network.get_gradients()
        
        if self.network.optimizer is None:
            raise TrainingError("Network not compiled with optimizer")
        
        self.network.optimizer.update(params, grads)
        
        return loss_value

    def _validate(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        y_pred = self.network.predict(X)
        
        if self.network.loss is None:
            raise TrainingError("Network not compiled with loss function")
        
        return np.mean(self.network.loss.forward(y_pred, y))

    def add_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)

    def get_history(self) -> Dict[str, List[float]]:
        return self.history

    def reset(self) -> None:
        self.history = {"loss": [], "val_loss": []}
        self.early_stopping_counter = 0
        self.best_loss = float('inf')
