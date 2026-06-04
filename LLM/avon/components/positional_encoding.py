import numpy as np


class PositionalEncoding:
    def __init__(self, d_model: int, max_seq_length: int = 512):
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]
