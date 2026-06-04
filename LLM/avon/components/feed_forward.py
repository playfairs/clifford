import numpy as np
from typing import Callable


class FeedForward:
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.dot(x, self.W1) + self.b1
        hidden = self._apply_activation(hidden)
        
        if self.dropout > 0:
            dropout_mask = (np.random.rand(*hidden.shape) > self.dropout).astype(float)
            hidden = hidden * dropout_mask / (1 - self.dropout)
        
        output = np.dot(hidden, self.W2) + self.b2
        return output
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "gelu":
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            return x
