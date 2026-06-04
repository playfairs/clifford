import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from dataclasses import dataclass

from clifford.core.types import ActivationFunction


class Activation(ABC):
    @abstractmethod
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class ReLU(Activation):
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0.0, x)

    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x > 0.0).astype(np.float64)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x > 0.0, x, self.alpha * x)

    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x > 0.0, 1.0, self.alpha)


class Sigmoid(Activation):
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        s = self.forward(x)
        return s * (1.0 - s)


class Tanh(Activation):
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tanh(x)

    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 - np.tanh(x) ** 2


class Softmax(Activation):
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        s = self.forward(x)
        return s * (1.0 - s)


class Linear(Activation):
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x

    def backward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(x)
