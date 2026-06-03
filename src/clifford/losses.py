import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from clifford.types import LossFunction
from clifford.exceptions import LossError


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class MeanSquaredError(Loss):
    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.mean(np.square(y_pred - y_true), axis=-1)

    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        n = y_pred.shape[0]
        return 2.0 * (y_pred - y_true) / n


class BinaryCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon

    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred), axis=-1)

    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        n = y_pred.shape[0]
        return (y_pred - y_true) / (n * y_pred * (1.0 - y_pred))


class CategoricalCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon

    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        n = y_pred.shape[0]
        return (y_pred - y_true) / n


class Hinge(Loss):
    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.mean(np.maximum(0.0, 1.0 - y_true * y_pred), axis=-1)

    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        mask = (y_true * y_pred) < 1.0
        n = y_pred.shape[0]
        grad = np.zeros_like(y_pred)
        grad[mask] = -y_true[mask] / n
        return grad
