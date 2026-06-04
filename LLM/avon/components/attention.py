import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        batch_size, seq_len, d_model = x.shape
        
        Q = np.dot(x, self.W_q) + self.b_q
        K = np.dot(x, self.W_k) + self.b_k
        V = np.dot(x, self.W_v) + self.b_v
        
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        scores = np.dot(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        attention_weights = self._softmax(scores)
        
        if self.dropout > 0:
            dropout_mask = (np.random.rand(*attention_weights.shape) > self.dropout).astype(float)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        output = np.dot(attention_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        output = np.dot(output, self.W_o) + self.b_o
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
