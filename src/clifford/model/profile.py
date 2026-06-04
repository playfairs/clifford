from typing import Dict, Any, List, Optional
import time
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict

from clifford.model.network import Network
from clifford.core.exceptions import ModelNotFoundError


class ModelProfiler:
    @staticmethod
    def profile_layer_times(network: Network, X: NDArray[np.float64], iterations: int = 10) -> Dict[str, Any]:
        layer_times = defaultdict(list)
        
        for _ in range(iterations):
            for layer in network.layers:
                start = time.time()
                layer.forward(X, training=False)
                end = time.time()
                layer_times[type(layer).__name__].append(end - start)
        
        stats = {}
        for layer_name, times in layer_times.items():
            stats[layer_name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "total": sum(times),
            }
        
        return stats

    @staticmethod
    def profile_parameter_distribution(network: Network) -> Dict[str, Any]:
        params = network.get_params()
        
        stats = {}
        for key, value in params.items():
            stats[key] = {
                "mean": np.mean(value),
                "std": np.std(value),
                "min": np.min(value),
                "max": np.max(value),
                "median": np.median(value),
                "sparsity": np.sum(value == 0) / value.size,
            }
        
        return stats

    @staticmethod
    def profile_gradient_flow(network: Network, X: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, Any]:
        if not network.loss:
            return {"error": "Network has no loss function"}
        
        network.forward(X)
        loss = network.loss.forward(network.layers[-1].output, y)
        grad = network.loss.backward(network.layers[-1].output, y)
        
        gradient_stats = {
            "loss_value": loss,
            "gradient_mean": np.mean(grad),
            "gradient_std": np.std(grad),
            "gradient_max": np.max(grad),
            "gradient_min": np.min(grad),
            "gradient_norm": np.linalg.norm(grad),
        }
        
        return gradient_stats

    @staticmethod
    def comprehensive_profile(network: Network, X: NDArray[np.float64], y: Optional[NDArray[np.float64]] = None, iterations: int = 10) -> Dict[str, Any]:
        profile = {
            "layer_times": ModelProfiler.profile_layer_times(network, X, iterations),
            "parameter_distribution": ModelProfiler.profile_parameter_distribution(network),
        }
        
        if y is not None:
            profile["gradient_flow"] = ModelProfiler.profile_gradient_flow(network, X, y)
        
        return profile
