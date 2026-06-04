from typing import Dict, Any, List, Optional
import numpy as np
from numpy.typing import NDArray

from clifford.model.network import Network
from clifford.core.exceptions import ModelNotFoundError, LayerError


class ModelValidator:
    @staticmethod
    def validate_architecture(network: Network) -> Dict[str, Any]:
        issues = []
        warnings = []
        
        if not network.layers:
            issues.append("Network has no layers")
        
        if network.input_dim is None:
            issues.append("Network input dimension is not set")
        
        if not network.compiled:
            warnings.append("Network is not compiled")
        
        for i, layer in enumerate(network.layers):
            if not hasattr(layer, 'input_size') and i > 0:
                warnings.append(f"Layer {i} may not have input size set")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "layer_count": len(network.layers),
            "compiled": network.compiled,
            "input_dim": network.input_dim,
        }

    @staticmethod
    def validate_parameters(network: Network) -> Dict[str, Any]:
        issues = []
        warnings = []
        
        params = network.get_params()
        
        if not params:
            issues.append("Network has no parameters")
        
        for key, value in params.items():
            if np.isnan(value).any():
                issues.append(f"Parameter {key} contains NaN values")
            
            if np.isinf(value).any():
                issues.append(f"Parameter {key} contains Inf values")
            
            if np.abs(value).max() > 1e6:
                warnings.append(f"Parameter {key} has very large values")
            
            if np.abs(value).max() < 1e-10 and np.abs(value).max() > 0:
                warnings.append(f"Parameter {key} has very small values")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "parameter_count": len(params),
            "total_parameters": sum(p.size for p in params.values()),
        }

    @staticmethod
    def validate_forward_pass(network: Network, X: NDArray[np.float64]) -> Dict[str, Any]:
        issues = []
        warnings = []
        
        try:
            output = network.predict(X)
            
            if np.isnan(output).any():
                issues.append("Forward pass produced NaN values")
            
            if np.isinf(output).any():
                issues.append("Forward pass produced Inf values")
            
            if output.shape[0] != X.shape[0]:
                issues.append(f"Output batch size {output.shape[0]} doesn't match input batch size {X.shape[0]}")
            
        except Exception as e:
            issues.append(f"Forward pass failed: {str(e)}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    @staticmethod
    def validate_backward_pass(network: Network, X: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[str, Any]:
        issues = []
        warnings = []
        
        if not network.loss:
            issues.append("Network has no loss function")
            return {"valid": False, "issues": issues, "warnings": warnings}
        
        try:
            network.forward(X)
            loss = network.loss.forward(network.layers[-1].output, y)
            grads = network.loss.backward(network.layers[-1].output, y)
            
            if np.isnan(grads).any():
                issues.append("Backward pass produced NaN gradients")
            
            if np.isinf(grads).any():
                issues.append("Backward pass produced Inf gradients")
            
        except Exception as e:
            issues.append(f"Backward pass failed: {str(e)}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    @staticmethod
    def comprehensive_validation(network: Network, X: Optional[NDArray[np.float64]] = None, y: Optional[NDArray[np.float64]] = None) -> Dict[str, Any]:
        validation = {
            "architecture": ModelValidator.validate_architecture(network),
            "parameters": ModelValidator.validate_parameters(network),
        }
        
        if X is not None:
            validation["forward_pass"] = ModelValidator.validate_forward_pass(network, X)
        
        if X is not None and y is not None:
            validation["backward_pass"] = ModelValidator.validate_backward_pass(network, X, y)
        
        all_issues = []
        all_warnings = []
        
        for key, result in validation.items():
            all_issues.extend(result.get("issues", []))
            all_warnings.extend(result.get("warnings", []))
        
        return {
            "valid": len(all_issues) == 0,
            "issues": all_issues,
            "warnings": all_warnings,
            "details": validation,
        }
