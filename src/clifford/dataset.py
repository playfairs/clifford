import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Iterator
from pathlib import Path
import json

from clifford.utils import normalize_data, denormalize_data, one_hot_encode, one_hot_decode
from clifford.exceptions import DatasetError


class Dataset:
    def __init__(self, name: str, X: NDArray[np.float64], y: NDArray[np.float64]):
        self.name = name
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.samples = X.shape[0]
        self.classes = y.shape[1] if len(y.shape) > 1 else None
        self.normalization_params: Optional[dict] = None
        self.normalized: bool = False

    def normalize(self, method: str = "minmax") -> "Dataset":
        if self.normalized:
            raise DatasetError("Dataset already normalized")
        
        X_normalized, params = normalize_data(self.X, method)
        self.X = X_normalized
        self.normalization_params = params
        self.normalized = True
        return self

    def denormalize(self) -> "Dataset":
        if not self.normalized or self.normalization_params is None:
            raise DatasetError("Dataset not normalized")
        
        self.X = denormalize_data(self.X, self.normalization_params)
        self.normalization_params = None
        self.normalized = False
        return self

    def shuffle(self) -> "Dataset":
        indices = np.random.permutation(self.samples)
        self.X = self.X[indices]
        self.y = self.y[indices]
        return self

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        split_idx = int(self.samples * (1 - ratio))
        X_train, X_val = self.X[:split_idx], self.X[split_idx:]
        y_train, y_val = self.y[:split_idx], self.y[split_idx:]
        
        train_dataset = Dataset(f"{self.name}_train", X_train, y_train)
        val_dataset = Dataset(f"{self.name}_val", X_val, y_val)
        
        train_dataset.normalization_params = self.normalization_params
        train_dataset.normalized = self.normalized
        val_dataset.normalization_params = self.normalization_params
        val_dataset.normalized = self.normalized
        
        return train_dataset, val_dataset

    def batch(self, batch_size: int) -> Iterator[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        for i in range(0, self.samples, batch_size):
            end_idx = min(i + batch_size, self.samples)
            yield self.X[i:end_idx], self.y[i:end_idx]

    def save(self, path: Path) -> None:
        data = {
            "name": self.name,
            "X": self.X.tolist(),
            "y": self.y.tolist(),
            "features": self.features,
            "samples": self.samples,
            "classes": self.classes,
            "normalization_params": self.normalization_params,
            "normalized": self.normalized
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        with open(path, 'r') as f:
            data = json.load(f)
        
        X = np.array(data["X"], dtype=np.float64)
        y = np.array(data["y"], dtype=np.float64)
        
        dataset = cls(data["name"], X, y)
        dataset.normalization_params = data["normalization_params"]
        dataset.normalized = data["normalized"]
        
        return dataset

    def __len__(self) -> int:
        return self.samples

    def __repr__(self) -> str:
        return f"Dataset(name='{self.name}', samples={self.samples}, features={self.features}, classes={self.classes})"


class DataGenerator:
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self) -> Iterator[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield self.dataset.X[batch_indices], self.dataset.y[batch_indices]

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))


def generate_xor_dataset(samples: int = 1000) -> Dataset:
    X = np.random.randint(0, 2, size=(samples, 2)).astype(np.float64)
    y = np.logical_xor(X[:, 0].astype(bool), X[:, 1].astype(bool)).astype(np.float64).reshape(-1, 1)
    return Dataset("xor", X, y)


def generate_circle_dataset(samples: int = 1000, radius: float = 1.0) -> Dataset:
    X = np.random.uniform(-radius, radius, size=(samples, 2)).astype(np.float64)
    y = (np.linalg.norm(X, axis=1) < radius * 0.7).astype(np.float64).reshape(-1, 1)
    return Dataset("circle", X, y)


def generate_spiral_dataset(samples: int = 1000) -> Dataset:
    n = samples // 2
    r = np.linspace(0.1, 1.0, n)
    theta1 = np.linspace(0, 4 * np.pi, n)
    theta2 = np.linspace(np.pi, 5 * np.pi, n)
    
    X1 = np.column_stack([r * np.cos(theta1), r * np.sin(theta1)])
    X2 = np.column_stack([r * np.cos(theta2), r * np.sin(theta2)])
    
    X = np.vstack([X1, X2]).astype(np.float64)
    y = np.array([0] * n + [1] * n).astype(np.float64).reshape(-1, 1)
    
    return Dataset("spiral", X, y)
