from typing import List, Dict, Tuple, Optional
import os
import pickle
import regex as re
from .config import TokenizerConfig

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer with GPT-4 style regex splitting.
    Iteratively merges the most frequent pair of adjacent tokens.
    """
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self.min_freq = config.min_frequency
        self.pattern = config.pattern
        
        # Base vocabulary: Bytes (0-255)
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # Special Tokens
        self.eos_id = 256
        self.vocab[self.eos_id] = b"<EOS>"
        
        # Merges start after bytes + special
        self.merge_start_id = 257
        
    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
        
    def _merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
        
    def train(self, text: str) -> None:
        """
        Train the tokenizer on the provided text string.
        """
        print(f"[Tokenizer] Training BPE with vocab_size={self.vocab_size}...")
        
        # 1. Regex Split
        pat = re.compile(self.pattern)
        text_chunks = pat.findall(text)
        print(f"[Tokenizer] Regex split: {len(text_chunks)} chunks")
        
        # 2. Encode chunks to bytes
        # ids_list is a list of lists of integers
        ids_list = [list(c.encode("utf-8")) for c in text_chunks]
        
        num_merges = self.vocab_size - self.merge_start_id
        if num_merges <= 0:
            print(f"[Tokenizer] Vocabulary size <= {self.merge_start_id}, no merges needed.")
            return

        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
            
            if not stats:
                break
                
            # Find most frequent pair
            # Tie-breaking: usually by freq, then by pair literal value?
            # We just take max freq.
            pair = None
            
            # Simple greedy max, but let's check constraints if we want
            # We must find the BEST valid pair
            candidates = sorted(stats.items(), key=lambda item: item[1], reverse=True)
            
            for candidate_pair, freq in candidates:
                if freq < self.min_freq:
                    continue
                    
                # Check length constraint to avoid overly long tokens?
                # Optional: p0_len + p1_len > 16 constraint from original code
                p0_len = len(self.vocab[candidate_pair[0]])
                p1_len = len(self.vocab[candidate_pair[1]])
                if p0_len + p1_len > 16:
                    continue
                
                pair = candidate_pair
                break
            
            if pair is None:
                print(f"[Tokenizer] Stopping early at vocab size {self.merge_start_id+i}: No valid pairs found.")
                break
            
            idx = self.merge_start_id + i
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # Apply merge to all chunks
            # In a real efficient implementation we would track locations, 
            # but for "Mini" GPT this is acceptable.
            new_ids_list = []
            for ids in ids_list:
                new_ids_list.append(self._merge(ids, pair, idx))
            ids_list = new_ids_list
            
            if (i+1) % 50 == 0:
                print(f"[Tokenizer] Merge {i+1}/{num_merges}: {pair} -> {idx} (Freq: {stats[pair]})")
                
        print(f"[Tokenizer] Training Complete. Final vocab size: {256 + len(self.merges)}")
        
    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs."""
        pat = re.compile(self.pattern)
        text_chunks = pat.findall(text)
        
        all_ids = []
        
        for chunk in text_chunks:
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                stats = self._get_stats(ids)
                # Find the pair with smallest merge index (priority)
                # This is equivalent to finding the pair that was merged earliest?
                # Actually, during inference, we should merge the pair that appears in merges with the lowest index?
                # Or simply the one that exists in merges.
                # Standard BPE: find pair with smallest rank (value in merges).
                
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                
                if pair not in self.merges:
                    break
                
                idx = self.merges[pair]
                ids = self._merge(ids, pair, idx)
            all_ids.extend(ids)
            
        return all_ids
        
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        tokens = b"".join([self.vocab[idx] for idx in ids if idx in self.vocab])
        return tokens.decode("utf-8", errors="replace")
        
    def save(self, path: str = "tokenizer.model") -> None:
        """Saves the tokenizer model (vocabulary and merges) to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'merges': self.merges,
                'vocab': self.vocab,
                'config': {
                    'vocab_size': self.vocab_size,
                    'min_frequency': self.min_freq,
                    'pattern': self.pattern
                }
            }, f)
        print(f"[Tokenizer] Saved model to {path}")
        
    def load(self, path: str = "tokenizer.model") -> None:
        """Loads the tokenizer model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer model not found at {path}")
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.merges = data['merges']
            self.vocab = data['vocab']
            # Load config if available
            if 'config' in data:
                conf = data['config']
                self.vocab_size = conf.get('vocab_size', self.vocab_size)
                self.config.vocab_size = self.vocab_size
        print(f"[Tokenizer] Loaded model from {path}")
