from typing import Optional
import copy

from clifford.model.network import Network
from clifford.model.layers import BaseLayer
from clifford.core.exceptions import ModelNotFoundError


class ModelCloner:
    @staticmethod
    def clone(network: Network, name: Optional[str] = None) -> Network:
        cloned_network = Network(name=name or f"{network.name}_clone")
        
        for layer in network.layers:
            if isinstance(layer, BaseLayer):
                cloned_layer = copy.deepcopy(layer)
                cloned_network.add(cloned_layer)
        
        if network.loss:
            cloned_network.loss = copy.deepcopy(network.loss)
        
        if network.optimizer:
            cloned_network.optimizer = copy.deepcopy(network.optimizer)
        
        cloned_network.compiled = network.compiled
        cloned_network.input_dim = network.input_dim
        
        return cloned_network

    @staticmethod
    def clone_with_weights(network: Network, name: Optional[str] = None) -> Network:
        cloned_network = ModelCloner.clone(network, name)
        cloned_network.set_params(network.get_params())
        return cloned_network
