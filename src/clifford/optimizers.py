import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Optional, Dict

from clifford.types import Optimizer
from clifford.exceptions import OptimizerError


class BaseOptimizer(ABC):
    @abstractmethod
    def update(self, params: Dict[str, NDArray[np.float64]], grads: Dict[str, NDArray[np.float64]]) -> None:
        pass


class SGD(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, params: Dict[str, NDArray[np.float64]], grads: Dict[str, NDArray[np.float64]]) -> None:
        for key in params:
            if key in grads:
                params[key] -= self.learning_rate * grads[key]


class Momentum(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities: Optional[Dict[str, NDArray[np.float64]]] = None

    def update(self, params: Dict[str, NDArray[np.float64]], grads: Dict[str, NDArray[np.float64]]) -> None:
        if self.velocities is None:
            self.velocities = {key: np.zeros_like(value) for key, value in params.items()}
        
        for key in params:
            if key in grads:
                self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grads[key]
                params[key] += self.velocities[key]


class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache: Optional[Dict[str, NDArray[np.float64]]] = None

    def update(self, params: Dict[str, NDArray[np.float64]], grads: Dict[str, NDArray[np.float64]]) -> None:
        if self.cache is None:
            self.cache = {key: np.zeros_like(value) for key, value in params.items()}
        
        for key in params:
            if key in grads:
                self.cache[key] = self.decay_rate * self.cache[key] + (1.0 - self.decay_rate) * np.square(grads[key])
                params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)


class Adam(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[Dict[str, NDArray[np.float64]]] = None
        self.v: Optional[Dict[str, NDArray[np.float64]]] = None
        self.t: int = 0

    def update(self, params: Dict[str, NDArray[np.float64]], grads: Dict[str, NDArray[np.float64]]) -> None:
        self.t += 1
        
        if self.m is None:
            self.m = {key: np.zeros_like(value) for key, value in params.items()}
            self.v = {key: np.zeros_like(value) for key, value in params.items()}
        
        for key in params:
            if key in grads:
                self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * np.square(grads[key])
                
                m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)
                
                params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class Adagrad(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache: Optional[Dict[str, NDArray[np.float64]]] = None

    def update(self, params: Dict[str, NDArray[np.float64]], grads: Dict[str, NDArray[np.float64]]) -> None:
        if self.cache is None:
            self.cache = {key: np.zeros_like(value) for key, value in params.items()}
        
        for key in params:
            if key in grads:
                self.cache[key] += np.square(grads[key])
                params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
