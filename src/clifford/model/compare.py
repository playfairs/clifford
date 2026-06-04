from typing import Dict, Any, List, Optional
import numpy as np
from numpy.typing import NDArray

from clifford.model.network import Network
from clifford.core.exceptions import ModelNotFoundError


class ModelComparator:
    @staticmethod
    def compare_architecture(network1: Network, network2: Network) -> Dict[str, Any]:
        return {
            "same_layer_count": len(network1.layers) == len(network2.layers),
            "network1_layers": len(network1.layers),
            "network2_layers": len(network2.layers),
            "same_input_dim": network1.input_dim == network2.input_dim,
            "network1_input_dim": network1.input_dim,
            "network2_input_dim": network2.input_dim,
            "same_loss": str(type(network1.loss)) == str(type(network2.loss)) if network1.loss and network2.loss else False,
            "same_optimizer": str(type(network1.optimizer)) == str(type(network2.optimizer)) if network1.optimizer and network2.optimizer else False,
        }

    @staticmethod
    def compare_parameters(network1: Network, network2: Network) -> Dict[str, Any]:
        params1 = network1.get_params()
        params2 = network2.get_params()
        
        param_diffs = {}
        for key in params1.keys():
            if key in params2:
                diff = np.abs(params1[key] - params2[key]).sum()
                param_diffs[key] = diff
        
        return {
            "same_param_keys": set(params1.keys()) == set(params2.keys()),
            "network1_param_count": len(params1),
            "network2_param_count": len(params2),
            "parameter_differences": param_diffs,
            "total_difference": sum(param_diffs.values()),
        }

    @staticmethod
    def compare_performance(network1: Network, network2: Network, X: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, Any]:
        loss1 = network1.evaluate(X, y)
        loss2 = network2.evaluate(X, y)
        
        return {
            "network1_loss": loss1,
            "network2_loss": loss2,
            "loss_difference": loss1 - loss2,
            "better_network": "network1" if loss1 < loss2 else "network2",
        }

    @staticmethod
    def compare_models(network1: Network, network2: Network, X: Optional[NDArray[np.float64]] = None, y: Optional[NDArray[np.float64]] = None) -> Dict[str, Any]:
        comparison = {
            "architecture": ModelComparator.compare_architecture(network1, network2),
            "parameters": ModelComparator.compare_parameters(network1, network2),
        }
        
        if X is not None and y is not None:
            comparison["performance"] = ModelComparator.compare_performance(network1, network2, X, y)
        
        return comparison
