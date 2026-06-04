from typing import Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray

from clifford.model.network import Network
from clifford.core.exceptions import ModelNotFoundError


class ModelQuantizer:
    @staticmethod
    def quantize_to_int8(network: Network) -> Network:
        import copy
        quantized_network = copy.deepcopy(network)
        
        params = quantized_network.get_params()
        for key, value in params.items():
            scale = np.max(np.abs(value)) / 127
            quantized = np.round(value / scale).astype(np.int8)
            params[key] = quantized.astype(np.float64) * scale
        
        quantized_network.set_params(params)
        return quantized_network

    @staticmethod
    def quantize_to_int16(network: Network) -> Network:
        import copy
        quantized_network = copy.deepcopy(network)
        
        params = quantized_network.get_params()
        for key, value in params.items():
            scale = np.max(np.abs(value)) / 32767
            quantized = np.round(value / scale).astype(np.int16)
            params[key] = quantized.astype(np.float64) * scale
        
        quantized_network.set_params(params)
        return quantized_network

    @staticmethod
    def estimate_quantization_error(network: Network, X: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, Any]:
        original_loss = network.evaluate(X, y)
        
        int8_network = ModelQuantizer.quantize_to_int8(network)
        int8_loss = int8_network.evaluate(X, y)
        
        int16_network = ModelQuantizer.quantize_to_int16(network)
        int16_loss = int16_network.evaluate(X, y)
        
        return {
            "original_loss": original_loss,
            "int8_loss": int8_loss,
            "int16_loss": int16_loss,
            "int8_error": abs(int8_loss - original_loss),
            "int16_error": abs(int16_loss - original_loss),
            "int8_error_percentage": (abs(int8_loss - original_loss) / original_loss) * 100 if original_loss > 0 else 0,
            "int16_error_percentage": (abs(int16_loss - original_loss) / original_loss) * 100 if original_loss > 0 else 0,
        }
