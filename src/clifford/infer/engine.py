from typing import Optional, Dict, Any, List, Iterator
import numpy as np
from numpy.typing import NDArray
from collections import OrderedDict
import time

from clifford.model import Network
from clifford.core.exceptions import ModelNotFoundError


class InferenceEngine:
    def __init__(self, network: Network):
        self.network = network
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = 100
        self.debug_mode = False
        self.fast_mode = False

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.debug_mode:
            print(f"[DEBUG] InferenceEngine.predict: Input shape {X.shape}")
            start = time.time()
            output = self.network.predict(X)
            end = time.time()
            print(f"[DEBUG] InferenceEngine.predict: Output shape {output.shape}")
            print(f"[DEBUG] InferenceEngine.predict: Time {end - start:.4f}s")
            return output
        else:
            return self.network.predict(X)

    def predict_batch(self, X: NDArray[np.float64], batch_size: int = 32) -> NDArray[np.float64]:
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            batch_pred = self.predict(batch)
            predictions.append(batch_pred)
        
        return np.vstack(predictions)

    def predict_streaming(self, X: Iterator[NDArray[np.float64]]) -> Iterator[NDArray[np.float64]]:
        for batch in X:
            yield self.predict(batch)

    def predict_cached(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        cache_key = hash(X.tobytes())
        
        if cache_key in self.cache:
            if self.debug_mode:
                print(f"[DEBUG] InferenceEngine.predict_cached: Cache hit")
            return self.cache[cache_key]
        
        output = self.predict(X)
        
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[cache_key] = output
        
        if self.debug_mode:
            print(f"[DEBUG] InferenceEngine.predict_cached: Cache miss, cached result")
        
        return output

    def predict_fast(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.fast_mode:
            return self.network.predict(X)
        return self.predict(X)

    def predict_debug(self, X: NDArray[np.float64]) -> Dict[str, Any]:
        original_debug = self.debug_mode
        self.debug_mode = True
        
        output = self.predict(X)
        
        self.debug_mode = original_debug
        
        return {
            "output": output,
            "input_shape": X.shape,
            "output_shape": output.shape,
            "network_layers": len(self.network.layers),
            "network_compiled": self.network.compiled,
        }

    def set_cache_size(self, size: int) -> None:
        self.cache_size = size
        while len(self.cache) > size:
            self.cache.popitem(last=False)

    def clear_cache(self) -> None:
        self.cache.clear()

    def enable_debug_mode(self) -> None:
        self.debug_mode = True

    def disable_debug_mode(self) -> None:
        self.debug_mode = False

    def enable_fast_mode(self) -> None:
        self.fast_mode = True

    def disable_fast_mode(self) -> None:
        self.fast_mode = False

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache),
            "cache_limit": self.cache_size,
            "cache_hit_rate": 0.0,
        }
