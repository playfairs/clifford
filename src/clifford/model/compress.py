from typing import Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray

from clifford.model.network import Network
from clifford.core.exceptions import ModelNotFoundError


class ModelCompressor:
    @staticmethod
    def calculate_sparsity(network: Network) -> Dict[str, float]:
        params = network.get_params()
        sparsity = {}
        
        for key, value in params.items():
            sparsity[key] = np.sum(value == 0) / value.size
        
        sparsity["overall"] = np.mean(list(sparsity.values()))
        
        return sparsity

    @staticmethod
    def prune_by_threshold(network: Network, threshold: float) -> Network:
        import copy
        pruned_network = copy.deepcopy(network)
        
        params = pruned_network.get_params()
        for key, value in params.items():
            mask = np.abs(value) > threshold
            params[key] = value * mask
        
        pruned_network.set_params(params)
        return pruned_network

    @staticmethod
    def prune_by_percentage(network: Network, percentage: float) -> Network:
        params = network.get_params()
        
        all_values = np.concatenate([value.flatten() for value in params.values()])
        threshold = np.percentile(np.abs(all_values), percentage * 100)
        
        return ModelCompressor.prune_by_threshold(network, threshold)

    @staticmethod
    def estimate_compression_ratio(network: Network) -> Dict[str, Any]:
        sparsity = ModelCompressor.calculate_sparsity(network)
        
        original_size = sum(p.nbytes for p in network.get_params().values())
        compressed_size = original_size * (1 - sparsity["overall"])
        
        return {
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": original_size / compressed_size if compressed_size > 0 else float('inf'),
            "space_saved_bytes": original_size - compressed_size,
            "space_saved_percentage": sparsity["overall"] * 100,
        }
