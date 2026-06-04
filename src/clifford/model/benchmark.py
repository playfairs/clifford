from typing import Dict, Any, List, Optional, Callable
import time
import numpy as np
from numpy.typing import NDArray

from clifford.model.network import Network
from clifford.core.exceptions import ModelNotFoundError


class ModelBenchmark:
    @staticmethod
    def benchmark_inference(network: Network, X: NDArray[np.float64], iterations: int = 100) -> Dict[str, Any]:
        times = []
        
        for _ in range(iterations):
            start = time.time()
            network.predict(X)
            end = time.time()
            times.append(end - start)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "median_time": np.median(times),
            "total_time": sum(times),
            "throughput": len(X) / np.mean(times),
        }

    @staticmethod
    def benchmark_training(network: Network, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int = 10) -> Dict[str, Any]:
        from clifford.train import Trainer
        
        trainer = Trainer(network)
        start = time.time()
        trainer.train(X, y, epochs=epochs, verbose=False)
        end = time.time()
        
        total_time = end - start
        
        return {
            "total_time": total_time,
            "time_per_epoch": total_time / epochs,
            "epochs_per_second": epochs / total_time,
            "final_loss": trainer.history["loss"][-1] if trainer.history["loss"] else None,
        }

    @staticmethod
    def benchmark_memory(network: Network) -> Dict[str, Any]:
        import sys
        
        params = network.get_params()
        total_params = sum(p.size for p in params.values())
        param_memory = sum(p.nbytes for p in params.values())
        
        network_size = sys.getsizeof(network)
        
        return {
            "total_parameters": total_params,
            "parameter_memory_bytes": param_memory,
            "parameter_memory_mb": param_memory / (1024 * 1024),
            "network_size_bytes": network_size,
            "network_size_mb": network_size / (1024 * 1024),
        }

    @staticmethod
    def comprehensive_benchmark(network: Network, X: NDArray[np.float64], y: NDArray[np.float64], inference_iterations: int = 100, training_epochs: int = 10) -> Dict[str, Any]:
        return {
            "inference": ModelBenchmark.benchmark_inference(network, X, inference_iterations),
            "training": ModelBenchmark.benchmark_training(network, X, y, training_epochs),
            "memory": ModelBenchmark.benchmark_memory(network),
        }
