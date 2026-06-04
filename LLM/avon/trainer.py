import numpy as np
from typing import Optional, Callable, List
from pathlib import Path

from avon.config import AvonConfig
from avon.model import Avon


class AvonTrainer:
    def __init__(self, model: Avon, config: AvonConfig, learning_rate: float = 1e-4):
        self.model = model
        self.config = config
        self.learning_rate = learning_rate
        
        self.global_step = 0
        self.checkpoint_dir = Path("LLM/avon/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_losses = []
    
    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray) -> float:
        logits, _ = self.model.forward(input_ids)
        
        loss = self._compute_loss(logits, target_ids)
        
        self._backward_pass(logits, target_ids)
        
        self.global_step += 1
        self.training_losses.append(loss)
        
        return loss
    
    def _compute_loss(self, logits: np.ndarray, target_ids: np.ndarray) -> float:
        batch_size, seq_len, vocab_size = logits.shape
        
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)
        
        shifted_logits = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        log_probs = -np.log(softmax_probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        loss = np.mean(log_probs)
        
        return loss
    
    def _backward_pass(self, logits: np.ndarray, target_ids: np.ndarray) -> None:
        batch_size, seq_len, vocab_size = logits.shape
        
        d_logits = np.zeros_like(logits)
        
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)
        
        shifted_logits = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        d_logits_flat = softmax_probs.copy()
        d_logits_flat[np.arange(len(targets_flat)), targets_flat] -= 1
        d_logits_flat = d_logits_flat / batch_size
        
        d_logits = d_logits_flat.reshape(batch_size, seq_len, vocab_size)
        
        d_lm_head = np.dot(d_logits.reshape(-1, vocab_size).T, self.model.norm.gamma.reshape(-1, 1).T).T
        d_lm_head = d_lm_head.reshape(self.model.lm_head.shape)
        
        d_norm_output = np.dot(d_logits, self.model.lm_head.T)
        
        d_norm_gamma = np.sum(d_norm_output, axis=(0, 1))
        d_norm_beta = np.sum(d_norm_output, axis=(0, 1))
        
        self.model.lm_head -= self.learning_rate * d_lm_head
        self.model.norm.gamma -= self.learning_rate * d_norm_gamma * 0.01
        self.model.norm.beta -= self.learning_rate * d_norm_beta * 0.01
        
        d_embedding = d_norm_output
        d_embedding_matrix = np.sum(d_embedding, axis=(0, 1))
        self.model.embedding.embedding_matrix -= self.learning_rate * d_embedding_matrix * 0.01
        
        for block in self.model.transformer_blocks:
            self._update_transformer_block_simple(block, d_norm_output)
    
    def _update_transformer_block_simple(self, block, d_output: np.ndarray) -> None:
        d_attn_output = d_output
        
        d_attn_Wo = np.sum(d_attn_output, axis=(0, 1))
        block.attention.W_o -= self.learning_rate * d_attn_Wo * 0.01
        
        d_attn_input = np.dot(d_attn_output, block.attention.W_o.T)
        
        d_attn_V = d_attn_input
        d_attn_Wv = np.sum(d_attn_V, axis=(0, 1))
        block.attention.W_v -= self.learning_rate * d_attn_Wv * 0.01
        
        d_attn_K = d_attn_input
        d_attn_Wk = np.sum(d_attn_K, axis=(0, 1))
        block.attention.W_k -= self.learning_rate * d_attn_Wk * 0.01
        
        d_attn_Q = d_attn_input
        d_attn_Wq = np.sum(d_attn_Q, axis=(0, 1))
        block.attention.W_q -= self.learning_rate * d_attn_Wq * 0.01
        
        d_ff_output = d_output
        d_ff_W2 = np.sum(d_ff_output, axis=(0, 1))
        block.feed_forward.W2 -= self.learning_rate * d_ff_W2 * 0.01
        
        d_ff_input = d_ff_output
        d_ff_W1 = np.sum(d_ff_input, axis=(0, 1))
        block.feed_forward.W1 -= self.learning_rate * d_ff_W1 * 0.01
    
    def train(self, input_ids: List[np.ndarray], target_ids: List[np.ndarray], epochs: int = 10, batch_size: int = 4) -> List[float]:
        all_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(0, len(input_ids), batch_size):
                batch_input = input_ids[i:i+batch_size]
                batch_target = target_ids[i:i+batch_size]
                
                if len(batch_input) < batch_size:
                    continue
                
                loss = self.train_step(np.array(batch_input), np.array(batch_target))
                epoch_losses.append(loss)
                
                if self.global_step % 10 == 0:
                    self.save_checkpoint()
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            all_losses.extend(epoch_losses)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return all_losses
    
    def save_checkpoint(self, path: Optional[str] = None) -> None:
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_{self.global_step}.npz"
        
        np.savez(
            path,
            W_q=self.model.transformer_blocks[0].attention.W_q,
            W_k=self.model.transformer_blocks[0].attention.W_k,
            W_v=self.model.transformer_blocks[0].attention.W_v,
            W_o=self.model.transformer_blocks[0].attention.W_o,
            embedding=self.model.embedding.embedding_matrix,
            lm_head=self.model.lm_head,
            global_step=self.global_step,
            training_losses=np.array(self.training_losses)
        )
    
    def load_checkpoint(self, path: str) -> None:
        data = np.load(path)
        
        for block in self.model.transformer_blocks:
            block.attention.W_q = data['W_q']
            block.attention.W_k = data['W_k']
            block.attention.W_v = data['W_v']
            block.attention.W_o = data['W_o']
        
        self.model.embedding.embedding_matrix = data['embedding']
        self.model.lm_head = data['lm_head']
        self.global_step = int(data['global_step'])
        
        if 'training_losses' in data:
            self.training_losses = list(data['training_losses'])

