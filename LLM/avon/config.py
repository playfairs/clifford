from dataclasses import dataclass
from typing import Optional


@dataclass
class AvonConfig:
    vocab_size: int = 50000
    max_seq_length: int = 512
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False
    
    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
