from typing import Optional, Dict, Any, List
import numpy as np
from numpy.typing import NDArray

from clifford.model.network import Network
from clifford.model.layers import BaseLayer
from clifford.core.exceptions import ModelNotFoundError, LayerError


class ModelMerger:
    @staticmethod
    def average_merge(networks: List[Network], name: str = "merged_network") -> Network:
        if not networks:
            raise ModelNotFoundError("No networks provided for merging")
        
        first_network = networks[0]
        merged_network = Network(name=name)
        
        for layer in first_network.layers:
            if isinstance(layer, BaseLayer):
                merged_network.add(layer)
        
        merged_params = {}
        for param_name in first_network.get_params().keys():
            param_values = [network.get_params()[param_name] for network in networks]
            averaged_param = np.mean(param_values, axis=0)
            merged_params[param_name] = averaged_param
        
        merged_network.set_params(merged_params)
        
        if first_network.loss:
            merged_network.loss = first_network.loss
        
        if first_network.optimizer:
            merged_network.optimizer = first_network.optimizer
        
        merged_network.compiled = first_network.compiled
        merged_network.input_dim = first_network.input_dim
        
        return merged_network

    @staticmethod
    def weighted_merge(networks: List[Network], weights: List[float], name: str = "merged_network") -> Network:
        if not networks or len(networks) != len(weights):
            raise ModelNotFoundError("Networks and weights must have same length")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise LayerError("Weights must sum to 1.0")
        
        first_network = networks[0]
        merged_network = Network(name=name)
        
        for layer in first_network.layers:
            if isinstance(layer, BaseLayer):
                merged_network.add(layer)
        
        merged_params = {}
        for param_name in first_network.get_params().keys():
            weighted_param = np.zeros_like(first_network.get_params()[param_name])
            for network, weight in zip(networks, weights):
                weighted_param += weight * network.get_params()[param_name]
            merged_params[param_name] = weighted_param
        
        merged_network.set_params(merged_params)
        
        if first_network.loss:
            merged_network.loss = first_network.loss
        
        if first_network.optimizer:
            merged_network.optimizer = first_network.optimizer
        
        merged_network.compiled = first_network.compiled
        merged_network.input_dim = first_network.input_dim
        
        return merged_network
