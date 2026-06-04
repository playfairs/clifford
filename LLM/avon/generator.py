import numpy as np
from typing import List, Optional

from avon.model import Avon


class AvonGenerator:
    def __init__(self, model: Avon, tokenizer: Optional['Tokenizer'] = None):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> np.ndarray:
        generated = input_ids.copy()
        
        for _ in range(max_length):
            if generated.shape[0] >= self.model.config.max_seq_length:
                generated = generated[-self.model.config.max_seq_length:]
            
            logits, _ = self.model.forward(generated[np.newaxis, :])
            logits = logits[0, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                logits = self._apply_top_k(logits, top_k)
            
            if top_p is not None:
                logits = self._apply_top_p(logits, top_p)
            
            if do_sample:
                probs = self._softmax(logits)
                next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(logits)
            
            generated = np.append(generated, next_token)
        
        return generated
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided. Cannot generate text from string prompt.")
        
        input_ids = self.tokenizer.encode(prompt)
        generated_ids = self.generate(input_ids, max_length, temperature, top_k, top_p)
        return self.tokenizer.decode(generated_ids)
    
    def _apply_top_k(self, logits: np.ndarray, k: int) -> np.ndarray:
        indices = np.argpartition(logits, -k)[-k:]
        mask = np.zeros_like(logits)
        mask[indices] = 1
        return logits * mask - 1e9 * (1 - mask)
    
    def _apply_top_p(self, logits: np.ndarray, p: float) -> np.ndarray:
        sorted_logits = np.sort(logits)[::-1]
        cumulative_probs = np.cumsum(self._softmax(sorted_logits))
        
        sorted_indices = np.argsort(logits)[::-1]
        sorted_indices_to_remove = cumulative_probs > p
        
        if sorted_indices_to_remove.any():
            sorted_indices_to_remove[1:] = np.logical_or(sorted_indices_to_remove[:-1], sorted_indices_to_remove[1:])
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -1e9
        
        return logits
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class Tokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_built = False
    
    def build_vocab(self, texts: List[str]) -> None:
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            self.word_to_id[word] = i + 2
            self.id_to_word[i + 2] = word
        
        self.word_to_id["<PAD>"] = 0
        self.id_to_word[0] = "<PAD>"
        self.word_to_id["<UNK>"] = 1
        self.id_to_word[1] = "<UNK>"
        
        self.vocab_built = True
    
    def encode(self, text: str) -> np.ndarray:
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        words = text.split()
        ids = []
        for word in words:
            ids.append(self.word_to_id.get(word, 1))
        return np.array(ids)
    
    def decode(self, ids: np.ndarray) -> str:
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        words = []
        for id in ids:
            word = self.id_to_word.get(int(id), "<UNK>")
            if word != "<PAD>":
                words.append(word)
        return " ".join(words)
