import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Any, Dict
import json

from clifford.layers import BaseLayer, Dense, Input
from clifford.losses import Loss, MeanSquaredError
from clifford.optimizers import BaseOptimizer, SGD
from clifford.exceptions import LayerError, OptimizerError


class Network:
    def __init__(self, name: str = "network"):
        self.name = name
        self.layers: List[BaseLayer] = []
        self.loss: Optional[Loss] = None
        self.optimizer: Optional[BaseOptimizer] = None
        self.compiled: bool = False
        self.input_dim: Optional[int] = None

    def add(self, layer: BaseLayer) -> None:
        if isinstance(layer, Input):
            self.input_dim = layer.input_dim
        self.layers.append(layer)

    def compile(self, loss: Loss, optimizer: BaseOptimizer) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.compiled = True

    def forward(self, x: NDArray[np.float64], training: bool = True) -> NDArray[np.float64]:
        if not self.layers:
            raise LayerError("No layers in network")
        
        output = x
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad: NDArray[np.float64]) -> None:
        if not self.layers:
            raise LayerError("No layers in network")
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_params(self) -> Dict[str, NDArray[np.float64]]:
        params: Dict[str, NDArray[np.float64]] = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for key, value in layer_params.items():
                params[f"layer_{i}_{key}"] = value
        return params

    def set_params(self, params: Dict[str, NDArray[np.float64]]) -> None:
        for i, layer in enumerate(self.layers):
            layer_params = {}
            for key in layer.get_params().keys():
                param_key = f"layer_{i}_{key}"
                if param_key in params:
                    layer_params[key] = params[param_key]
            if layer_params:
                layer.set_params(layer_params)

    def get_gradients(self) -> Dict[str, NDArray[np.float64]]:
        grads: Dict[str, NDArray[np.float64]] = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_gradients'):
                layer_grads = layer.get_gradients()
                for key, value in layer_grads.items():
                    grads[f"layer_{i}_{key}"] = value
        return grads

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(x, training=False)

    def evaluate(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, float]:
        y_pred = self.predict(x)
        if self.loss is None:
            raise OptimizerError("Network not compiled with loss function")
        
        loss_value = np.mean(self.loss.forward(y_pred, y))
        return {"loss": loss_value}

    def summary(self) -> str:
        lines = [f"Network: {self.name}", "=" * 50]
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            params = layer.get_params()
            param_count = sum(p.size for p in params.values())
            total_params += param_count
            
            if isinstance(layer, Dense):
                lines.append(f"Layer {i}: {layer_name} (units={layer.units}, params={param_count})")
            elif isinstance(layer, Input):
                lines.append(f"Layer {i}: {layer_name} (input_dim={layer.input_dim})")
            else:
                lines.append(f"Layer {i}: {layer_name} (params={param_count})")
        
        lines.append("=" * 50)
        lines.append(f"Total parameters: {total_params}")
        return "\n".join(lines)

    def get_architecture(self) -> Dict[str, Any]:
        architecture = {
            "name": self.name,
            "layers": []
        }
        
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer_info = {
                    "type": "Dense",
                    "units": layer.units,
                    "input_dim": layer.input_dim,
                    "activation": layer.activation.__class__.__name__
                }
            elif isinstance(layer, Input):
                layer_info = {
                    "type": "Input",
                    "input_dim": layer.input_dim
                }
            else:
                layer_info = {
                    "type": layer.__class__.__name__
                }
            architecture["layers"].append(layer_info)
        
        return architecture

    def save_architecture(self, path: str) -> None:
        architecture = self.get_architecture()
        with open(path, 'w') as f:
            json.dump(architecture, f, indent=2)

    @classmethod
    def from_architecture(cls, architecture: Dict[str, Any]) -> "Network":
        network = cls(name=architecture["name"])
        
        for layer_info in architecture["layers"]:
            if layer_info["type"] == "Input":
                network.add(Input(input_dim=layer_info["input_dim"]))
            elif layer_info["type"] == "Dense":
                from clifford.activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear
                
                activation_map = {
                    "ReLU": ReLU(),
                    "Sigmoid": Sigmoid(),
                    "Tanh": Tanh(),
                    "Softmax": Softmax(),
                    "LeakyReLU": LeakyReLU(),
                    "Linear": Linear()
                }
                
                activation = activation_map.get(layer_info["activation"], ReLU())
                network.add(Dense(
                    units=layer_info["units"],
                    input_dim=layer_info.get("input_dim"),
                    activation=activation
                ))
        
        return network
