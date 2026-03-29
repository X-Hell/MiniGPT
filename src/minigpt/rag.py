import numpy as np
import os
import re
import pickle
from typing import List, Tuple, Optional
from .model import MiniTransformer
from .tokenizer import BPETokenizer

class VectorStore:
    """
    A lightweight, strictly NumPy-based vector store for on-device RAG.
    Stores embeddings and their corresponding text chunks.
    """
    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[str] = []
        
    def add(self, vectors: np.ndarray, chunks: List[str]):
        """
        Add vectors and chunks to the store.
        vectors: (N, D) array
        chunks: list of N strings
        """
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.concatenate([self.vectors, vectors], axis=0)
        self.chunks.extend(chunks)
        
    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for top-k similar chunks using cosine similarity.
        query_vector: (D,) or (1, D) array
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []
            
        # Normalize vectors for cosine similarity (if not already)
        # Assuming query and stored vectors might not be normalized.
        # Ideally, we normalize on add/encode, but safety first.
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 1e-9:
            query_vector = query_vector / q_norm
            
        # Compute dot product: (N, D) @ (D,) -> (N,)
        scores = np.dot(self.vectors, query_vector.flatten())
        
        # Norms of stored vectors (can be cached for speed)
        v_norms = np.linalg.norm(self.vectors, axis=1)
        # Avoid division by zero
        v_norms[v_norms < 1e-9] = 1.0
        
        scores = scores / v_norms
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
            
        return results
        
    def save(self, path: str):
        """Save store to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'vectors': self.vectors, 'chunks': self.chunks}, f)
            
    def load(self, path: str):
        """Load store from disk."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.chunks = data['chunks']


class QueryRefiner:
    """
    On-device lightweight query refinement using regex and heuristics.
    No heavy NLP dependencies (no spacy, no nltk).
    """
    def __init__(self):
        # Small lexicon for synonym expansion (extendable)
        self.synonyms = {
            "build": ["create", "construct", "make", "develop"],
            "robot": ["automaton", "machine", "bot", "droid"],
            "learn": ["study", "understand", "acquire", "master"],
            "code": ["program", "script", "software", "implementation"],
            "fast": ["quick", "rapid", "speedy", "swift"],
            "error": ["bug", "issue", "problem", "fault"],
            "train": ["teach", "educate", "instruct", "optimize"],
            "model": ["network", "system", "architecture", "LLM"]
        }
        
    def refine(self, query: str) -> str:
        """
        Refine the query by extracting keywords and adding synonyms.
        """
        # 1. Basic cleaning
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        words = clean_query.split()
        
        # 2. Extract potential keywords (len > 3)
        keywords = [w for w in words if len(w) > 3]
        
        # 3. Synonym Expansion
        expanded = []
        for w in keywords:
            if w in self.synonyms:
                # Add up to 2 synonyms
                expanded.extend(self.synonyms[w][:2])
                
        if not expanded:
            return query # No refinement possible
            
        # Construct refined query
        # "Original query + synonyms"
        refined = query + " " + " ".join(expanded)
        print(f"[Refiner] '{query}' -> '{refined}'")
        return refined

class Retriever:
    """
    Handles document ingestion, embedding generation (using MiniTransformer),
    and retrieval orchestration with dense scoring.
    """
    def __init__(self, model: MiniTransformer, tokenizer: BPETokenizer, vector_store: VectorStore):
        self.model = model
        self.tokenizer = tokenizer
        self.store = vector_store
        self.refiner = QueryRefiner()
        
    def encode(self, text_list: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings using the model.
        Strategy: Mean pooling of the final layer hidden states.
        """
        embeddings = []
        for text in text_list:
            # Tokenize
            tokens = self.tokenizer.encode(text)
            if len(tokens) == 0:
                continue
                
            # Truncate if too long (simple approach for now)
            max_len = self.model.config.max_len
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                
            token_ids = np.array(tokens)[np.newaxis, :] # (1, T)
            
            # Forward pass (no training, no cache update needed ideally, but model api requires it)
            # We just want the final hidden states.
            # MiniTransformer.forward returns (logits, attn_weights)
            # But we need the hidden states (x_final).
            # We can access `model.cache_x_final` after forward.
            
            self.model.forward(token_ids)
            
            # (1, T, D)
            hidden_states = self.model.cache_x_final
            
            # Mean Pooling: (1, D)
            embedding = np.mean(hidden_states, axis=1)
            embeddings.append(embedding)
            
        if not embeddings:
            return np.array([])
            
        return np.concatenate(embeddings, axis=0) # (N, D)
        
    def index_documents(self, text_chunks: List[str]):
        """
        Embed and store a list of text chunks.
        """
        print(f"[Retriever] Indexing {len(text_chunks)} chunks...")
        
        # Dense index
        vectors = self.encode(text_chunks)
        if len(vectors) > 0:
            self.store.add(vectors, text_chunks)
        
        print(f"[Retriever] Indexed {len(vectors)} vectors.")
        
    def retrieve(self, query: str, k: int = 8) -> List[Tuple[str, float]]:
        """
        Retrieve relevant context for a query (dense only).
        """
        q_vec = self.encode([query])
        if len(q_vec) == 0:
            return []
            
        return self.store.search(q_vec[0], k=k)
        
    def retrieve_with_refinement(self, query: str, k: int = 8) -> dict:
        """
        Orchestrate retrieval with optional refinement.
        Returns detailed result dict.
        """
        # 1. Initial Retrieve
        results = self.retrieve(query, k=k)
        
        top_score = 0.0
        if results:
            top_score = results[0][1]
            
        refinement_triggered = False
        
        # 2. Check Decision Gate (< 0.50)
        if top_score < 0.50:
            refinement_triggered = True
            refined_query = self.refiner.refine(query)
            
            if refined_query != query:
                refined_results = self.retrieve(refined_query, k=k)
                
                # Check if improved
                if refined_results and refined_results[0][1] >= 0.50:
                    results = refined_results
                    top_score = results[0][1]
                elif refined_results and refined_results[0][1] > top_score:
                    # Even if not > 0.5, if it's better, take it? 
                    # Req says: "If new top score >= 0.50 use retrieved context; else proceed"
                    # We will stick to the strict rule: if < 0.5 still, we fail later.
                    # But we can update results to be the "best effort" ones.
                    results = refined_results
                    top_score = results[0][1]

        return {
            "results": results, 
            "top_score": top_score,
            "refinement_triggered": refinement_triggered
        }
