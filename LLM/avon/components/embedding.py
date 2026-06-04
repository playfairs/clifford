import numpy as np


class Embedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding_matrix = np.random.randn(vocab_size, d_model) * 0.02
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        return self.embedding_matrix[input_ids]
