from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TokenizerConfig:
    vocab_size: int = 1024

class MiniTokenizer:
    """
    A simple character-level tokenizer that maps characters to integer IDs.
    Iterates over the ASCII range and reserves special tokens.
    """
    def __init__(self, config: TokenizerConfig):
        self.vocab_size = config.vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Reserved tokens
        self.pad_token_id = 0
        self.id_to_char[0] = "<PAD>"
        
        self.unknown_token_id = 1
        self.id_to_char[1] = "<UNK>"
        
        # Populate vocab with printable ASCII
        # Starting from 2
        idx = 2
        
        # Standard ASCII printable characters
        import string
        chars = string.printable # digits, ascii_letters, punctuation, whitespace
        
        for char in chars:
            if idx >= self.vocab_size:
                break
            if char not in self.char_to_id:
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
                idx += 1
                
        print(f"[Tokenizer] Initialized with {len(self.char_to_id)} characters. Max vocab: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        ids = []
        for char in text:
            ids.append(self.char_to_id.get(char, self.unknown_token_id))
        return ids

    def decode(self, ids: List[int]) -> str:
        text = []
        for i in ids:
            val = self.id_to_char.get(i, "")
            if val == "<UNK>" or val == "<PAD>":
                continue # Skip special tokens for cleaner output in demo
            text.append(val)
        return "".join(text)

    def size_bytes(self):
        # Rough estimate only
        import sys
        return sys.getsizeof(self.char_to_id) + sys.getsizeof(self.id_to_char)
