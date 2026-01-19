from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import pickle

@dataclass
class TokenizerConfig:
    vocab_size: int = 1024
    min_frequency: int = 2

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer.
    Iteratively merges the most frequent pair of adjacent tokens.
    """
    def __init__(self, config: TokenizerConfig):
        self.vocab_size = config.vocab_size
        self.min_freq = config.min_frequency
        
        # Base vocabulary: Bytes (0-255)
        # We represent tokens as integers. 0-255 are literals.
        # Merges will define new tokens starting from 256.
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # Special tokens
        self.special_tokens = {
            "<PAD>": 0, # Map to NULL byte for simplicity or handle separately? 
                        # Ideally, BPE starts after specials.
                        # Let's reserve 256+ for merges, and keep 0-255 as bytes.
                        # Actually, standard BPE usually maps bytes to unicode chars, 
                        # but for "from scratch" simplicity, let's keep it raw.
        }
        
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
        
    def train(self, text: str):
        print(f"[Tokenizer] Training BPE with vocab_size={self.vocab_size}...")
        
        # Convert text to UTF-8 bytes -> integers
        # This is our initial token stream
        ids = list(text.encode("utf-8"))
        print(f"[Tokenizer] Initial tokens (bytes): {len(ids)}")
        
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
                
            # Find most frequent pair that satisfies length constraint
            pair = None
            while stats:
                candidate = max(stats, key=stats.get)
                
                # Check constraints
                if stats[candidate] < self.min_freq:
                    # Best remaining is too rare, stop everything
                    pair = None
                    break
                    
                # Check length
                p0_len = len(self.vocab[candidate[0]])
                p1_len = len(self.vocab[candidate[1]])
                if p0_len + p1_len > 16:
                    # Too long, remove and try next best
                    del stats[candidate]
                    continue
                else:
                    # Valid!
                    pair = candidate
                    break
            
            if pair is None:
                print("[Tokenizer] Stopping early: No valid pairs found (freq/len constraints).")
                break
            
            # Constraint: Don't merge if resulting token is too long (Mega-Token Fix)
            # Check length of the merged candidate
            # We need to look up what the new token would be.
            # It's vocab[pair[0]] + vocab[pair[1]]
            p0_bytes = self.vocab[pair[0]]
            p1_bytes = self.vocab[pair[1]]
            new_len = len(p0_bytes) + len(p1_bytes)
            
            if new_len > 16:
                # Skip this pair, it's too long.
                # Problem: 'stats' contains it as max. If we just continue, we'll pick it again.
                # We must effectively 'ban' this pair for this iteration or remove it from consideration.
                # Simple fix: discard this pair from stats and try next best?
                # Or just ignore it and break? No, there might be other valid pairs.
                # Let's delete from stats and re-pick.
                del stats[pair]
                # Since we found the max and it was bad, we need to find the *next* max.
                # Re-sorting stats every time is slow but safe for this script.
                if not stats:
                    break
                pair = max(stats, key=stats.get)
                # Re-check new pair (recursion/loop needed theoretically, but let's just do a while loop)
                # Let's refactor slightly to handle this.
                pass # Logic handled by loop below
            
            idx = 256 + i
            # Record merge
            self.merges[pair] = idx
            
            # Update vocab mapping for decoding
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # Apply merge to current data
            ids = self._merge(ids, pair, idx)
            
            if (i+1) % 50 == 0:
                print(f"[Tokenizer] Merge {i+1}/{num_merges}: {pair} -> {idx} (Freq: {stats[pair]})")
                
        print(f"[Tokenizer] Training Complete. Final vocab size: {256 + len(self.merges)}")
        print(f"[Tokenizer] Compression: {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens")

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find the pair with the lowest index in self.merges (earliest merge)
            # If multiple pairs exist in stats that are in merges, we must merge the one 
            # that was created *first* (lowest index value)? 
            # Actually, we just greedily merge whatever is available in priority order.
            
            # In standard BPE inference, we iterate through merges in order.
            # But that is O(N * M).
            # Optimization: 
            # Look at all pairs in current sequence. find the one with min index in merges.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # No more mergeable pairs
            
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
            
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = b"".join([self.vocab[idx] for idx in ids if idx in self.vocab]) # Skip unknowns
        # errors='replace' to handle incomplete utf-8 sequences
        return tokens.decode("utf-8", errors="replace")
        
    def save(self, path="tokenizer.model"):
        with open(path, 'wb') as f:
            pickle.dump({
                'merges': self.merges,
                'vocab': self.vocab,
                'config': {'vocab_size': self.vocab_size}
            }, f)
        print(f"[Tokenizer] Saved model to {path}")
        
    def load(self, path="tokenizer.model"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.merges = data['merges']
            self.vocab = data['vocab']
            self.vocab_size = data['config']['vocab_size']
        print(f"[Tokenizer] Loaded model from {path}")

