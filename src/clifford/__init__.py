from clifford.network import Network
from clifford.layers import Dense, Input
from clifford.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Linear
from clifford.losses import MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy, Hinge
from clifford.optimizers import SGD, Momentum, RMSProp, Adam, Adagrad
from clifford.trainer import Trainer
from clifford.dataset import Dataset, DataGenerator, generate_xor_dataset, generate_circle_dataset, generate_spiral_dataset
from clifford.metrics import Accuracy, Precision, Recall, F1Score, MeanAbsoluteError, MeanSquaredErrorMetric, RootMeanSquaredError, R2Score, MetricsCollector
from clifford.persistence import ModelPersistence
from clifford.registry import ModelRegistry
from clifford.search import SearchEngine
from clifford.config import Config, ConfigManager, TrainingConfig, DatabaseConfig, ModelConfig, GUIConfig
from clifford.types import TrainingConfig as TrainingConfigType, ModelMetadata, TrainingRun, Checkpoint, DatasetInfo, MetricRecord
from clifford.exceptions import CliffordError, ModelNotFoundError, LayerError, ShapeMismatchError, OptimizerError, LossError, DatabaseError, ConfigurationError, TrainingError, DatasetError, PersistenceError, SearchError

__version__ = "0.1.0"
__all__ = [
    "Network",
    "Dense",
    "Input",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Linear",
    "MeanSquaredError",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "Hinge",
    "SGD",
    "Momentum",
    "RMSProp",
    "Adam",
    "Adagrad",
    "Trainer",
    "Dataset",
    "DataGenerator",
    "generate_xor_dataset",
    "generate_circle_dataset",
    "generate_spiral_dataset",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "MeanAbsoluteError",
    "MeanSquaredErrorMetric",
    "RootMeanSquaredError",
    "R2Score",
    "MetricsCollector",
    "ModelPersistence",
    "ModelRegistry",
    "SearchEngine",
    "Config",
    "ConfigManager",
    "TrainingConfig",
    "DatabaseConfig",
    "ModelConfig",
    "GUIConfig",
    "TrainingConfigType",
    "ModelMetadata",
    "TrainingRun",
    "Checkpoint",
    "DatasetInfo",
    "MetricRecord",
    "CliffordError",
    "ModelNotFoundError",
    "LayerError",
    "ShapeMismatchError",
    "OptimizerError",
    "LossError",
    "DatabaseError",
    "ConfigurationError",
    "TrainingError",
    "DatasetError",
    "PersistenceError",
    "SearchError",
    "__version__",
]
