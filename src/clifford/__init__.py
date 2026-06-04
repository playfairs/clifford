from clifford.model import Network, Dense, Input, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Linear
from clifford.model import MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy, Hinge
from clifford.model import SGD, Momentum, RMSProp, Adam, Adagrad
from clifford.train import Trainer
from clifford.data import Dataset, DataGenerator, generate_xor_dataset, generate_circle_dataset, generate_spiral_dataset
from clifford.metrics import Accuracy, Precision, Recall, F1Score, MeanAbsoluteError, MeanSquaredErrorMetric, RootMeanSquaredError, R2Score, MetricsCollector
from clifford.store import Store
from clifford.database import Registry
from clifford.search import Search
from clifford.config import Config, ConfigManager, TrainingConfig, DatabaseConfig, ModelConfig, GUIConfig
from clifford.core import ModelMetadata, TrainingRun, Checkpoint, DatasetInfo, MetricRecord
from clifford.core import CliffordError, ModelNotFoundError, LayerError, ShapeMismatchError, OptimizerError, LossError, DatabaseError, ConfigurationError, TrainingError, DatasetError, PersistenceError, SearchError

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
    "Store",
    "Registry",
    "Search",
    "Config",
    "ConfigManager",
    "TrainingConfig",
    "DatabaseConfig",
    "ModelConfig",
    "GUIConfig",
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
