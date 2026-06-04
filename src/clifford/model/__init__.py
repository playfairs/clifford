from clifford.model.network import Network
from clifford.model.layers import Dense, Input
from clifford.model.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Linear
from clifford.model.losses import MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy, Hinge
from clifford.model.optimizers import SGD, Momentum, RMSProp, Adam, Adagrad
from clifford.model.version import VersionManager, ModelVersion
from clifford.model.clone import ModelCloner
from clifford.model.merge import ModelMerger
from clifford.model.compare import ModelComparator
from clifford.model.benchmark import ModelBenchmark
from clifford.model.validate import ModelValidator
from clifford.model.profile import ModelProfiler
from clifford.model.compress import ModelCompressor
from clifford.model.quantize import ModelQuantizer
from clifford.model.sentience import EmergentPersonality, SelfModel, EmergentConsciousness, ConsciousnessManager

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
    "VersionManager",
    "ModelVersion",
    "ModelCloner",
    "ModelMerger",
    "ModelComparator",
    "ModelBenchmark",
    "ModelValidator",
    "ModelProfiler",
    "ModelCompressor",
    "ModelQuantizer",
    "EmergentPersonality",
    "SelfModel",
    "EmergentConsciousness",
    "ConsciousnessManager",
]
