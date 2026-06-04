import numpy as np
from typing import Optional, Tuple

from avon.config import AvonConfig
from avon.components.embedding import Embedding
from avon.components.positional_encoding import PositionalEncoding
from avon.components.transformer_block import TransformerBlock


class Avon:
    def __init__(self, config: AvonConfig):
        self.config = config
        
        self.embedding = Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        self.transformer_blocks = [
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                config.activation
            )
            for _ in range(config.n_layers)
        ]
        
        self.norm = LayerNorm(config.d_model)
        
        self.lm_head = np.random.randn(config.d_model, config.vocab_size) * 0.02
        self.lm_head_bias = np.zeros(config.vocab_size)
    
    def forward(self, input_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x = self.embedding.forward(input_ids)
        x = self.pos_encoding.forward(x)
        
        all_attentions = []
        for block in self.transformer_blocks:
            x, attn_weights = block.forward(x, mask)
            if self.config.output_attentions:
                all_attentions.append(attn_weights)
        
        x = self.norm.forward(x)
        
        logits = np.dot(x, self.lm_head) + self.lm_head_bias
        
        return logits, all_attentions if self.config.output_attentions else None
    
    def generate(self, input_ids: np.ndarray, max_length: int = 100, temperature: float = 1.0) -> np.ndarray:
        generated = input_ids.copy()
        
        for _ in range(max_length):
            current_length = len(generated)
            if current_length >= self.config.max_seq_length:
                generated = generated[-self.config.max_seq_length:]
                current_length = len(generated)
            
            try:
                logits, _ = self.forward(generated[np.newaxis, :])
                logits = logits[0, -1, :] / temperature
                
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                
                next_token = np.random.choice(len(probs), p=probs)
                generated = np.append(generated, next_token)
            except Exception as e:
                print(f"Generation error: {e}")
                break
        
        return generated


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
