import numpy as np


class LayerNorm:
    def __init__(self, d_model: int, epsilon: float = 1e-5):
        self.d_model = d_model
        self.epsilon = epsilon
        
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta
