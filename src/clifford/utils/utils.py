import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List
import numpy as np
from numpy.typing import NDArray


def get_project_root() -> Path:
    env_path = os.environ.get("CLIFFORD_PROJECT_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return current.parents[3]


def get_asset_path() -> Path:
    env_path = os.environ.get("CLIFFORD_ASSET_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return get_project_root() / "assets" / "clifford.png"


def get_db_path() -> Path:
    env_path = os.environ.get("CLIFFORD_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return get_project_root() / "database" / "clifford.db"


def get_legacy_db_paths() -> List[Path]:
    root = get_project_root()
    return [
        root / "src" / "database" / "clifford.db",
    ]


def get_models_dir() -> Path:
    env_path = os.environ.get("CLIFFORD_MODELS_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return get_project_root() / "models"


def get_legacy_model_dirs() -> List[Path]:
    root = get_project_root()
    return [
        root / "src" / "models",
    ]


def ensure_directory(path: Path) -> None:
    target = path if not path.suffix else path.parent
    target.mkdir(parents=True, exist_ok=True)


def serialize_ndarray(arr: NDArray[np.float64]) -> Dict[str, Any]:
    return {
        "data": arr.tolist(),
        "shape": arr.shape,
        "dtype": str(arr.dtype)
    }


def deserialize_ndarray(data: Dict[str, Any]) -> NDArray[np.float64]:
    return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])


def one_hot_encode(labels: NDArray[np.int64], num_classes: int) -> NDArray[np.float64]:
    encoded = np.zeros((labels.size, num_classes), dtype=np.float64)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded


def one_hot_decode(encoded: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.argmax(encoded, axis=1)


def shuffle_data(X: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def split_data(X: NDArray[np.float64], y: NDArray[np.float64], split_ratio: float) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    split_idx = int(len(X) * (1 - split_ratio))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return X_train, X_val, y_train, y_val


def normalize_data(X: NDArray[np.float64], method: str = "minmax") -> Tuple[NDArray[np.float64], Dict[str, Any]]:
    if method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        normalized = (X - min_val) / range_val
        params = {"min": min_val.tolist(), "max": max_val.tolist(), "method": "minmax"}
    elif method == "zscore":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        normalized = (X - mean) / std
        params = {"mean": mean.tolist(), "std": std.tolist(), "method": "zscore"}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return normalized, params


def denormalize_data(X: NDArray[np.float64], params: Dict[str, Any]) -> NDArray[np.float64]:
    if params["method"] == "minmax":
        min_val = np.array(params["min"])
        max_val = np.array(params["max"])
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        return X * range_val + min_val
    elif params["method"] == "zscore":
        mean = np.array(params["mean"])
        std = np.array(params["std"])
        return X * std + mean
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


def create_batches(X: NDArray[np.float64], y: NDArray[np.float64], batch_size: int) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    num_samples = len(X)
    batches = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batches.append((X[i:end_idx], y[i:end_idx]))
    return batches


def calculate_fan_in_fan_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) == 2:
        fan_in, fan_out = shape[1], shape[0]
    elif len(shape) == 4:
        fan_in = shape[1] * shape[2] * shape[3]
        fan_out = shape[0] * shape[2] * shape[3]
    else:
        fan_in = fan_out = np.prod(shape[1:])
    return int(fan_in), int(fan_out)


def he_uniform(shape: Tuple[int, ...]) -> NDArray[np.float64]:
    fan_in, _ = calculate_fan_in_fan_out(shape)
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape).astype(np.float64)


def he_normal(shape: Tuple[int, ...]) -> NDArray[np.float64]:
    fan_in, _ = calculate_fan_in_fan_out(shape)
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0.0, std, size=shape).astype(np.float64)


def xavier_uniform(shape: Tuple[int, ...]) -> NDArray[np.float64]:
    fan_in, fan_out = calculate_fan_in_fan_out(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float64)


def xavier_normal(shape: Tuple[int, ...]) -> NDArray[np.float64]:
    fan_in, fan_out = calculate_fan_in_fan_out(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, size=shape).astype(np.float64)
