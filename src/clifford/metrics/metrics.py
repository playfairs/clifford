import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        pass


class Accuracy(Metric):
    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = (y_true > 0.5).astype(int).flatten()
        
        if y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = (y_pred > 0.5).astype(int).flatten()
        
        return float(np.mean(y_true_labels == y_pred_labels))

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class Precision(Metric):
    def __init__(self, average: str = "binary"):
        self.average = average

    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)
            
            if self.average == "macro":
                precisions = []
                for cls in np.unique(y_true_labels):
                    true_positives = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
                    predicted_positives = np.sum(y_pred_labels == cls)
                    if predicted_positives > 0:
                        precisions.append(true_positives / predicted_positives)
                return float(np.mean(precisions)) if precisions else 0.0
            else:
                true_positives = np.sum((y_pred_labels == y_true_labels) & (y_true_labels == 1))
                predicted_positives = np.sum(y_pred_labels == 1)
                if predicted_positives > 0:
                    return float(true_positives / predicted_positives)
                return 0.0
        else:
            y_true_binary = (y_true > 0.5).astype(int).flatten()
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            
            true_positives = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            predicted_positives = np.sum(y_pred_binary == 1)
            
            if predicted_positives > 0:
                return float(true_positives / predicted_positives)
            return 0.0

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class Recall(Metric):
    def __init__(self, average: str = "binary"):
        self.average = average

    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)
            
            if self.average == "macro":
                recalls = []
                for cls in np.unique(y_true_labels):
                    true_positives = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
                    actual_positives = np.sum(y_true_labels == cls)
                    if actual_positives > 0:
                        recalls.append(true_positives / actual_positives)
                return float(np.mean(recalls)) if recalls else 0.0
            else:
                true_positives = np.sum((y_pred_labels == y_true_labels) & (y_true_labels == 1))
                actual_positives = np.sum(y_true_labels == 1)
                if actual_positives > 0:
                    return float(true_positives / actual_positives)
                return 0.0
        else:
            y_true_binary = (y_true > 0.5).astype(int).flatten()
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            
            true_positives = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            actual_positives = np.sum(y_true_binary == 1)
            
            if actual_positives > 0:
                return float(true_positives / actual_positives)
            return 0.0

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class F1Score(Metric):
    def __init__(self, average: str = "binary"):
        self.average = average
        self.precision = Precision(average=average)
        self.recall = Recall(average=average)

    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        precision = self.precision.compute(y_true, y_pred)
        recall = self.recall.compute(y_true, y_pred)
        
        if precision + recall > 0:
            return float(2 * precision * recall / (precision + recall))
        return 0.0

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class MeanAbsoluteError(Metric):
    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class MeanSquaredErrorMetric(Metric):
    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return float(np.mean(np.square(y_true - y_pred)))

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class RootMeanSquaredError(Metric):
    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(np.square(y_true - y_pred))))

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class R2Score(Metric):
    def compute(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        ss_res = np.sum(np.square(y_true - y_pred))
        ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
        
        if ss_tot > 0:
            return float(1 - ss_res / ss_tot)
        return 0.0

    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.compute(y_true, y_pred)


class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.history: Dict[str, List[float]] = {}

    def add_metric(self, name: str, metric: Metric) -> None:
        self.metrics[name] = metric
        self.history[name] = []

    def compute_all(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> Dict[str, float]:
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute(y_true, y_pred)
            results[name] = value
            self.history[name].append(value)
        return results

    def get_history(self) -> Dict[str, List[float]]:
        return self.history

    def reset(self) -> None:
        self.history = {name: [] for name in self.metrics.keys()}
