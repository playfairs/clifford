from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Optional, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ActivationFunction(Protocol):
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...


@runtime_checkable
class LossFunction(Protocol):
    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        ...


@runtime_checkable
class Optimizer(Protocol):
    def update(self, params: Dict[str, NDArray[np.float64]], grads: Dict[str, NDArray[np.float64]]) -> None:
        ...


@runtime_checkable
class Layer(Protocol):
    def forward(self, x: NDArray[np.float64], training: bool = True) -> NDArray[np.float64]:
        ...

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def get_params(self) -> Dict[str, NDArray[np.float64]]:
        ...

    def set_params(self, params: Dict[str, NDArray[np.float64]]) -> None:
        ...


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    shuffle: bool = True
    verbose: bool = True


@dataclass
class ModelMetadata:
    name: str
    architecture: str
    hyperparameters: Dict[str, Any]
    created_at: str
    updated_at: str


@dataclass
class TrainingRun:
    id: str
    model_id: str
    config: TrainingConfig
    start_time: str
    end_time: Optional[str]
    status: str
    final_loss: Optional[float]
    final_metrics: Optional[Dict[str, float]]


@dataclass
class Checkpoint:
    id: str
    run_id: str
    epoch: int
    loss: float
    metrics: Dict[str, float]
    timestamp: str
    path: str


@dataclass
class DatasetInfo:
    id: str
    name: str
    samples: int
    features: int
    classes: Optional[int]
    path: str
    created_at: str


@dataclass
class MetricRecord:
    id: str
    run_id: str
    epoch: int
    metric_name: str
    value: float
    timestamp: str


BatchMode = str


class BatchMode:
    FULL = "full"
    MINI_BATCH = "mini_batch"
    STOCHASTIC = "stochastic"
