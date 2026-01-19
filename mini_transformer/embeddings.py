import numpy as np
from .matmul import explicit_matmul

class EmbeddingLayer:
    def __init__(self, vocab_size, d_model, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Token Embeddings: Real-valued matrix
        # Initialize small random weights
        self.W_emb = np.random.normal(scale=0.02, size=(vocab_size, d_model)).astype(np.float32)
        
        # Positional Embeddings: Sinusoidal
        self.PE = self._build_positional_encoding(max_len, d_model)
        
        print(f"[Embeddings] Table: {self.W_emb.shape} | Mem: {self.W_emb.nbytes/1024:.2f} KB")

    def _build_positional_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def forward(self, token_ids):
        """
        token_ids: list or array of shape (seq_len,)
        returns: (seq_len, d_model)
        """
        seq_len = len(token_ids)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        # Token embedding lookup (can be viewed as matmul with one-hot, but usually direct indexing)
        # We'll stick to indexing for performance/standard practice, but we could visualize it.
        # Creating a temporary one-hot just for "explicit matmul" might be too heavy (vocab_size x seq_len)
        # So we just index.
        tokens_emb = self.W_emb[token_ids] # (seq_len, d_model)
        
        # Add positional encodings
        # We assume we are processing positions 0..seq_len-1 (ignoring offset for now if doing iterative req)
        # NOTE: For autoregressive inference step-by-step, we might need to know the current 'step' index.
        # But usually in training we pass whole seq. 
        # In inference, we might pass a single token.
        # Let's handle generic case: we need an offset if processing one token.
        # But for this simple implementation, let's assume we pass the full sequence so far, or handle it in `forward_step`.
        
        # Actually, let's make `forward` take an optional `start_pos`.
        pass
        
    def forward_seq(self, token_ids, start_pos=0):
        """
        token_ids: (T,)
        start_pos: int, used for PE slicing
        """
        T = len(token_ids)
        tokens_emb = self.W_emb[token_ids] # (T, D)
        
        # RoPE is now used in Attention, so we remove absolute PE here.
        # pos_emb = self.PE[start_pos : start_pos + T] # (T, D)
        
        return tokens_emb # + pos_emb
