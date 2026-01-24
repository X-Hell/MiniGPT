import numpy as np
import os
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


class BM25:
    """
    Simple BM25 implementation for lexical scoring.
    NumPy-only, no external dependencies.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict = {}  # term -> count of docs containing term
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.corpus: List[List[str]] = []
        self.N: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return re.sub(r'[^\w\s]', '', text.lower()).split()
    
    def index(self, docs: List[str]):
        """Build BM25 index from documents."""
        self.corpus = [self._tokenize(doc) for doc in docs]
        self.N = len(self.corpus)
        self.doc_lens = [len(doc) for doc in self.corpus]
        self.avg_dl = np.mean(self.doc_lens) if self.doc_lens else 1.0
        
        # Count document frequencies
        self.doc_freqs = {}
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
    
    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for query against a single document."""
        if doc_idx >= len(self.corpus):
            return 0.0
        
        query_terms = self._tokenize(query)
        doc = self.corpus[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        for term in query_terms:
            if term not in self.doc_freqs:
                continue
            
            # Term frequency in document
            tf = doc.count(term)
            if tf == 0:
                continue
            
            # IDF
            df = self.doc_freqs[term]
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
            score += idf * (numerator / denominator)
        
        return score
    
    def score_all(self, query: str) -> np.ndarray:
        """Score all documents for a query."""
        return np.array([self.score(query, i) for i in range(self.N)])

import re


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
    and retrieval orchestration with hybrid dense + BM25 scoring.
    """
    def __init__(self, model: MiniTransformer, tokenizer: BPETokenizer, vector_store: VectorStore):
        self.model = model
        self.tokenizer = tokenizer
        self.store = vector_store
        self.refiner = QueryRefiner()
        self.bm25 = BM25()  # Lexical index
        self._raw_chunks: List[str] = []  # Keep raw chunks for BM25
        
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
        Also builds BM25 index for hybrid retrieval.
        """
        print(f"[Retriever] Indexing {len(text_chunks)} chunks...")
        self._raw_chunks = text_chunks
        
        # Dense index
        vectors = self.encode(text_chunks)
        if len(vectors) > 0:
            self.store.add(vectors, text_chunks)
        
        # BM25 index
        self.bm25.index(text_chunks)
        print(f"[Retriever] Indexed {len(vectors)} vectors + BM25.")
        
    def retrieve(self, query: str, k: int = 8) -> List[Tuple[str, float]]:
        """
        Retrieve relevant context for a query (dense only).
        """
        q_vec = self.encode([query])
        if len(q_vec) == 0:
            return []
            
        return self.store.search(q_vec[0], k=k)
    
    def retrieve_hybrid(self, query: str, k: int = 6, alpha: float = 0.7) -> List[Tuple[str, float, int]]:
        """
        Hybrid retrieval combining dense and BM25 scores.
        Returns: List of (chunk, combined_score, chunk_idx)
        alpha: weight for dense score (1-alpha for BM25)
        """
        if not self._raw_chunks:
            return []
        
        # Dense scores
        dense_results = self.retrieve(query, k=len(self._raw_chunks))  # Get all
        dense_scores = {chunk: score for chunk, score in dense_results}
        
        # BM25 scores
        bm25_scores = self.bm25.score_all(query)
        # Normalize BM25 to 0-1
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Combine scores
        combined = []
        for idx, chunk in enumerate(self._raw_chunks):
            dense_s = dense_scores.get(chunk, 0.0)
            bm25_s = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            combined_s = alpha * dense_s + (1 - alpha) * bm25_s
            combined.append((chunk, combined_s, idx))
        
        # Sort by combined score and take top-k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]

        
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
