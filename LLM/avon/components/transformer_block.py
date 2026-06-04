import numpy as np
from typing import Optional

from avon.components.attention import MultiHeadAttention
from avon.components.feed_forward import FeedForward
from avon.components.layer_norm import LayerNorm


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple:
        attn_output, attn_weights = self.attention.forward(x, mask)
        x = self.norm1.forward(x + attn_output)
        
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)
        
        return x, attn_weights
