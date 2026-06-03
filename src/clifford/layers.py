import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Optional, Dict

from clifford.types import Layer
from clifford.activations import Activation, ReLU
from clifford.utils import he_uniform
from clifford.exceptions import LayerError, ShapeMismatchError


class BaseLayer(ABC):
    @abstractmethod
    def forward(self, x: NDArray[np.float64], training: bool = True) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, NDArray[np.float64]]:
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, NDArray[np.float64]]) -> None:
        pass


class Dense(BaseLayer):
    def __init__(self, units: int, input_dim: Optional[int] = None, activation: Optional[Activation] = None):
        self.units = units
        self.input_dim = input_dim
        self.activation = activation if activation else ReLU()
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[NDArray[np.float64]] = None
        self.input: Optional[NDArray[np.float64]] = None
        self.output: Optional[NDArray[np.float64]] = None
        self.z: Optional[NDArray[np.float64]] = None

    def initialize(self, input_dim: int) -> None:
        if self.input_dim is None:
            self.input_dim = input_dim
        elif self.input_dim != input_dim:
            raise ShapeMismatchError(f"Input dimension mismatch: expected {self.input_dim}, got {input_dim}")
        
        self.weights = he_uniform((self.input_dim, self.units))
        self.bias = np.zeros((1, self.units), dtype=np.float64)

    def forward(self, x: NDArray[np.float64], training: bool = True) -> NDArray[np.float64]:
        if self.weights is None or self.bias is None:
            self.initialize(x.shape[1])
        
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None or self.z is None or self.weights is None:
            raise LayerError("Forward pass must be called before backward pass")
        
        activation_grad = self.activation.backward(self.z)
        grad = grad * activation_grad
        
        self.d_weights = np.dot(self.input.T, grad)
        self.d_bias = np.sum(grad, axis=0, keepdims=True)
        
        return np.dot(grad, self.weights.T)

    def get_params(self) -> Dict[str, NDArray[np.float64]]:
        if self.weights is None or self.bias is None:
            raise LayerError("Layer not initialized")
        return {"weights": self.weights, "bias": self.bias}

    def set_params(self, params: Dict[str, NDArray[np.float64]]) -> None:
        if "weights" not in params or "bias" not in params:
            raise LayerError("Parameters must include 'weights' and 'bias'")
        self.weights = params["weights"]
        self.bias = params["bias"]
        self.input_dim = self.weights.shape[0]
        self.units = self.weights.shape[1]

    def get_gradients(self) -> Dict[str, NDArray[np.float64]]:
        if not hasattr(self, 'd_weights') or not hasattr(self, 'd_bias'):
            raise LayerError("Gradients not computed")
        return {"weights": self.d_weights, "bias": self.d_bias}


class Input(BaseLayer):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.input: Optional[NDArray[np.float64]] = None

    def forward(self, x: NDArray[np.float64], training: bool = True) -> NDArray[np.float64]:
        if x.shape[1] != self.input_dim:
            raise ShapeMismatchError(f"Input shape mismatch: expected {self.input_dim}, got {x.shape[1]}")
        self.input = x
        return x

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        return grad

    def get_params(self) -> Dict[str, NDArray[np.float64]]:
        return {}

    def set_params(self, params: Dict[str, NDArray[np.float64]]) -> None:
        pass
