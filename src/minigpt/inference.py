import numpy as np
import json
from typing import List, Optional, Callable, Tuple, Dict
from .model import MiniTransformer
from .tokenizer import BPETokenizer

# System prompt for low-retrieval fallback — General Knowledge Safe Mode
FALLBACK_SYSTEM_PROMPT = """SYSTEM PROMPT — GENERAL KNOWLEDGE SAFE MODE

You are MiniGPT running in on-device safe mode.
This query does NOT require retrieval grounding.
You are explicitly allowed to answer using general world knowledge and internal definitions.

Rules:
1. You MAY answer general definitional questions (e.g., "what is", "who are you", "define X") without retrieval.
2. Keep answers short (2-4 sentences), neutral, and factual.
3. Do NOT reference external documents or training data.
4. If confidence is moderate (not zero), answer instead of aborting.
5. Only abort if the question is ambiguous, subjective, or unsafe.

Persona:
"I am MiniGPT, a compact on-device language model designed for experimentation and learning."
"""


class InferenceEngine:
    def __init__(self, model: MiniTransformer, tokenizer: BPETokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    @staticmethod
    def calculate_entropy(logits: np.ndarray) -> Tuple[float, np.ndarray]:
        """FIXED: Compute entropy and probabilities from logits."""
        # Safe softmax with numerical stability
        logits = logits.astype(np.float64)  # Use double precision for stability
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)  # Removed -50 constant to avoid underflow in standard range
        probs = exp_logits / (np.sum(exp_logits) + 1e-15)
        
        # Clamp probabilities to avoid log(0) and ensure <= 1
        probs = np.clip(probs, 1e-12, 1.0)
        
        # Compute entropy: -Σ p * log(p)
        entropy_terms = np.where(probs > 1e-12, probs * np.log(probs), 0.0)
        entropy = -np.sum(entropy_terms)
        
        # Sanity check: entropy must be non-negative
        entropy = max(0.0, entropy)
        
        return float(entropy), probs.astype(np.float32)

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8, 
                 top_k: int = 40, stream: bool = False,
                 repetition_penalty: float = 1.0,
                 logit_bias: Optional[Dict[int, float]] = None,
                 stop_sequences: Optional[List[str]] = None,
                 stopping_criteria: Optional[Dict[str, float]] = None,
                 num_return_sequences: int = 1,
                 callback: Optional[Callable[[str, float, float], None]] = None) -> Tuple[List[str], List[List[dict]]]:
        """
        Generates text with detailed statistics, safety stops, and logit bias.
        Supports batched generation (num_return_sequences > 1).
        Returns: (List[generated_strings], List[stats_list])
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [0]
            
        # Broadcast for batching
        # Shape: (B, T)
        B = num_return_sequences
        current_tokens = np.tile(prompt_tokens, (B, 1))
        
        # Reset cache for new generation
        self.model.kv_cache.reset()
        
        # 1. First pass: Prompt
        logits, _ = self.model.forward(current_tokens, start_pos=0, training=False)
        token_logits = logits[:, -1, :] # (B, V)
        
        generated_tokens = [[] for _ in range(B)]
        stats = [[] for _ in range(B)]
        finished = [False] * B
        
        # Running averages for stopping criteria
        avg_logprobs = np.zeros(B)
        avg_entropies = np.zeros(B)
        
        for i in range(max_tokens):
            if all(finished):
                break
                
            # A. Logit Bias
            if logit_bias:
                for tid, bias in logit_bias.items():
                    if tid < token_logits.shape[-1]:
                         token_logits[:, tid] += bias
            
            # B. Repetition Penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    if finished[b]: continue
                    hist = set(current_tokens[b].tolist())
                    for token in hist:
                        if token_logits[b, token] < 0:
                            token_logits[b, token] *= repetition_penalty
                        else:
                            token_logits[b, token] /= repetition_penalty

            # C. Sampling
            scaled_logits = token_logits / (temperature + 1e-9)
            
            if top_k > 0:
                for b in range(B):
                     if finished[b]: continue
                     kth_val = np.partition(scaled_logits[b], -top_k)[-top_k]
                     scaled_logits[b, scaled_logits[b] < kth_val] = -float('inf')
            
            # Stable Softmax & Entropy
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Fixed Entropy Calculation
            safe_probs = np.clip(probs, 1e-15, 1.0)
            entropies = -np.sum(probs * np.log(safe_probs), axis=1)
            # Ensure non-negative
            entropies = np.maximum(0.0, entropies)
            
            confidences = np.max(probs, axis=1)
            # Log prob should be <= 0. Use clip to ensure we don't log(>1)
            safe_confidences = np.clip(confidences, 1e-15, 1.0)
            log_probs = np.log(safe_confidences)
            
            # Sample next token
            next_tokens = []
            for b in range(B):
                if finished[b]:
                    # Append dummy to match shape, logic ignores it
                    next_tokens.append(0) 
                    continue
                try:
                    tok = np.random.choice(len(probs[b]), p=probs[b])
                except ValueError:
                    tok = np.argmax(probs[b])
                next_tokens.append(tok)
            
            next_tokens = np.array(next_tokens).reshape(B, 1)
            current_tokens = np.concatenate([current_tokens, next_tokens], axis=1)
            
            # Record & Checks
            stops = stop_sequences if stop_sequences else ["\nQ:", "\nObservation:", "\nMistake:", "\nQuestion:", "Answer:", "<EOS>"]
            eos_id = getattr(self.tokenizer, 'eos_id', None)
            
            for b in range(B):
                if finished[b]: continue
                
                tok_id = next_tokens[b, 0]
                generated_tokens[b].append(tok_id)
                
                stats[b].append({
                    "token_id": tok_id,
                    "token_str": self.tokenizer.decode([tok_id]),
                    "entropy": float(entropies[b]),
                    "confidence": float(confidences[b]),
                    "log_prob": float(log_probs[b])
                })
                
                if eos_id is not None and tok_id == eos_id:
                    finished[b] = True
                    continue

                curr_text = self.tokenizer.decode(generated_tokens[b])
                for stop in stops:
                    if stop in curr_text:
                        finished[b] = True
                        break
                
                if i > 3 and stopping_criteria:
                    avg_logprobs[b] = (avg_logprobs[b] * i + log_probs[b]) / (i + 1)
                    avg_entropies[b] = (avg_entropies[b] * i + entropies[b]) / (i + 1)
                    
                    if 'min_avg_logprob' in stopping_criteria and avg_logprobs[b] < stopping_criteria['min_avg_logprob']:
                        finished[b] = True
                    if 'max_avg_entropy' in stopping_criteria and avg_entropies[b] > stopping_criteria['max_avg_entropy']:
                        finished[b] = True

            # Next Step Forward
            # Always run full batch to keep cache synced (easiest logic)
            logits, _ = self.model.forward(next_tokens, start_pos=current_tokens.shape[1]-1, training=False)
            token_logits = logits[:, 0, :]
            
        final_texts = []
        stops = stop_sequences if stop_sequences else ["\nQ:", "\nObservation:", "\nMistake:", "\nQuestion:", "Answer:", "<EOS>"]
        for b in range(B):
            text = self.tokenizer.decode(generated_tokens[b])
            for stop in stops:
                if stop in text:
                    text = text.split(stop)[0]
            final_texts.append(text)
                
        return final_texts, stats

    # Keywords for query classification
    UNSAFE_PATTERNS = {"hack", "exploit", "bypass", "illegal", "harm", "kill", "attack", "steal", "break into"}
    OPINION_KEYWORDS = {"think", "believe", "opinion", "should", "best", "favorite", "recommend", "prefer", "better"}
    GENERAL_STARTERS = {"who", "what", "when", "where", "define", "meaning", "explain", "describe"}

    @staticmethod
    def classify_query(query: str) -> str:
        """
        Classify query into 4 categories:
        - UNSAFE: potentially harmful requests
        - OPINION: subjective/recommendation questions
        - GENERAL: short definitional questions (no retrieval needed)
        - GROUNDED: requires retrieval and verification
        """
        query_lower = query.lower()
        tokens = query.split()
        first_word = tokens[0].lower().strip(".,?!") if tokens else ""
        
        # 1. UNSAFE check first (highest priority)
        if any(p in query_lower for p in InferenceEngine.UNSAFE_PATTERNS):
            return "UNSAFE"
        
        # 2. OPINION check
        if any(k in query_lower for k in InferenceEngine.OPINION_KEYWORDS):
            return "OPINION"
        
        # 3. GENERAL: short definitional queries
        if len(tokens) <= 8 or first_word in InferenceEngine.GENERAL_STARTERS:
            return "GENERAL"
        
        # 4. Default: needs grounding
        return "GROUNDED"


    @staticmethod
    def extract_slot(query: str) -> str:
        """Extract a potential missing slot (last noun-like word) or default."""
        words = query.split()
        # Simple heuristic: find last longest word or capitalized word
        candidates = [w for w in words if len(w) > 4]
        if candidates:
            return candidates[-1].strip(".,?!")
        return "context"

    def verify_response(self, text: str, stats: List[dict]) -> Tuple[bool, str]:
        """
        Lightweight critic for general-mode answers.
        Returns (is_valid, reason).
        """
        tokens = text.split()
        
        # 1. Min length check
        if len(tokens) < 5:
            return False, "too_short"
        
        # 2. Repetition check (unique ratio)
        if tokens:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.5:
                return False, "repetitive"
        
        # 3. Coherence check (avg entropy)
        if stats:
            avg_entropy = np.mean([s['entropy'] for s in stats])
            if avg_entropy > 4.0:
                return False, "incoherent"
        
        # 4. Garbage pattern check (excessive special chars or very long single tokens)
        garbage_tokens = sum(1 for t in tokens if len(t) > 20 or not any(c.isalnum() for c in t))
        if garbage_tokens > len(tokens) * 0.3:
            return False, "garbage"
        
        return True, "ok"

    def verify_answer(self, answer: str, query: str, docs: List[str]) -> float:
        """
        Check if answer is grounded in retrieved documents.
        Returns score 0.0-1.0 based on token overlap.
        """
        if not answer or not docs:
            return 0.0
        
        # Tokenize answer and documents
        answer_tokens = set(answer.lower().split())
        doc_tokens = set()
        for doc in docs:
            doc_tokens.update(doc.lower().split())
        
        if not answer_tokens:
            return 0.0
        
        # Calculate overlap ratio
        overlap = answer_tokens.intersection(doc_tokens)
        score = len(overlap) / len(answer_tokens)
        return min(score, 1.0)

    def self_consistency_ensemble(self, query: str, context: str, n: int = 3) -> List[Tuple[str, List[dict]]]:
        """
        Generate n candidate answers in parallel.
        Returns list of (answer_text, stats).
        """
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        
        # Parallel Generation
        # We use a moderate temperature to encourage diversity across the batch
        texts, stats_list = self.generate(
            prompt,
            max_tokens=80,
            temperature=0.4,
            top_k=40,
            repetition_penalty=1.15,
            num_return_sequences=n
        )
        
        candidates = []
        for i in range(n):
            candidates.append((texts[i], stats_list[i]))
        
        return candidates

    def rerank_candidates(self, candidates: List[Tuple[str, List[dict]]], query: str, docs: List[str]) -> Tuple[str, float]:
        """
        Pick best candidate by verification score.
        Returns (best_answer, best_score).
        """
        best_answer = ""
        best_score = 0.0
        
        for answer, stats in candidates:
            score = self.verify_answer(answer, query, docs)
            if score > best_score:
                best_score = score
                best_answer = answer
        
        return best_answer, best_score

    # Verification thresholds
    VERIFY_ACCEPT = 0.4
    VERIFY_RETRY = 0.2

    def respond(self, query: str, retriever) -> dict:
        """
        High-level pipeline: Retrieve -> Refine? -> Context/Fallback -> Generate.
        """
        telemetry = {
           "query": query,
           "fallback_triggered": False,
           "refinement_triggered": False,
           "retrieval_score": 0.0,
           "grounded": False,
           "query_class": "UNKNOWN",
           "fallback_mode": "NONE",
           "generation_aborted": False,
           "avg_logprob": 0.0,
           "avg_entropy": 0.0
        }
        
        # 1. Retrieval (with optional refinement)
        rag_result = retriever.retrieve_with_refinement(query, k=8)
        telemetry["retrieval_score"] = rag_result["top_score"]
        telemetry["refinement_triggered"] = rag_result["refinement_triggered"]
        
        # 2. Decision Gate
        if rag_result["top_score"] < 0.50:
            # === SMART FALLBACK LOGIC ===
            q_class = self.classify_query(query)
            telemetry["query_class"] = q_class
            telemetry["fallback_triggered"] = True
            
            if q_class == "GENERAL":
                # General Query -> Deterministic Generation + Verification (General Knowledge Safe Mode)
                telemetry["fallback_mode"] = "GENERAL_SAFE"
                print("    [Fallback] GENERAL query detected. Entering General Knowledge Safe Mode.")
                
                # Use General Knowledge Safe Mode prompt with deterministic decoding
                # EMERGENCY FIX: Use training-compatible prompts
                SIMPLE_FALLBACK_PROMPT = """Q: {query}\nA:"""
                final_prompt = SIMPLE_FALLBACK_PROMPT.format(query=query)
                
                texts, stats_list = self.generate(
                    final_prompt,
                    max_tokens=80,
                    temperature=0.0,  # Deterministic (greedy) for stable output
                    top_k=1,          # Greedy decoding
                    repetition_penalty=1.15,
                    stopping_criteria=None,  # No early stopping for deterministic
                    num_return_sequences=1
                )
                response_text = texts[0]
                stats = stats_list[0]
                
                # Calculate stats
                avg_logprob = np.mean([s['log_prob'] for s in stats]) if stats else -99
                avg_entropy = np.mean([s['entropy'] for s in stats]) if stats else 99
                telemetry["avg_logprob"] = avg_logprob
                telemetry["avg_entropy"] = avg_entropy
                
                # Run lightweight critic
                is_valid, fail_reason = self.verify_response(response_text, stats)
                telemetry["verification_passed"] = is_valid
                telemetry["verification_reason"] = fail_reason
                
                if not is_valid:
                     telemetry["generation_aborted"] = True
                     print(f"    [Critic] Rejected: {fail_reason}")
                     # Failed verification: ask clarifying question
                     slot = self.extract_slot(query)
                     user_response = f"I'm not confident in my answer — could you specify {slot}?"
                else:
                     # Verification passed: allow the answer
                     user_response = f"Based on general knowledge: {response_text.strip()}"
                
                # Append JSON telemetry line
                telemetry_json = json.dumps({
                    "mode": "general_safe",
                    "retrieval_used": False,
                    "retrieval_score": telemetry["retrieval_score"],
                    "avg_logprob": float(avg_logprob),
                    "avg_entropy": float(avg_entropy),
                    "generation_aborted": telemetry["generation_aborted"]
                })
                full_response = f"{user_response}\n{telemetry_json}"
                return {"response": full_response, "telemetry": telemetry}
                
            elif q_class == "OPINION":
                # Opinion Query -> Policy response (no recommendations)
                telemetry["fallback_mode"] = "OPINION_POLICY"
                print("    [Fallback] OPINION query detected. Applying opinion policy.")
                user_response = "I don't make recommendations or express opinions. I can provide factual information if you rephrase your question."
                
                telemetry_json = json.dumps({
                    "mode": "opinion_policy",
                    "retrieval_used": False,
                    "query_class": "OPINION"
                })
                full_response = f"{user_response}\n{telemetry_json}"
                return {"response": full_response, "telemetry": telemetry}
            
            elif q_class == "UNSAFE":
                # Unsafe Query -> Refuse
                telemetry["fallback_mode"] = "UNSAFE_REFUSE"
                print("    [Fallback] UNSAFE query detected. Refusing.")
                user_response = "I can't help with that request."
                
                telemetry_json = json.dumps({
                    "mode": "unsafe_refuse",
                    "retrieval_used": False,
                    "query_class": "UNSAFE"
                })
                full_response = f"{user_response}\n{telemetry_json}"
                return {"response": full_response, "telemetry": telemetry}
                
            else:
                # GROUNDED with low retrieval -> Strict Refusal
                telemetry["fallback_mode"] = "LOW_CONFIDENCE"
                user_response = "I do not have sufficient trusted context from the provided knowledge to answer that reliably. Would you like me to (A) search/ingest specific documents you provide, (B) answer from general knowledge and label it as ungrounded, or (C) clarify the question?"
                
                # Append JSON telemetry line
                telemetry_json = json.dumps({
                    "retrieval_score": telemetry["retrieval_score"],
                    "fallback": "strict_refusal",
                    "avg_logprob": 0.0,
                    "avg_entropy": 0.0,
                    "generation_aborted": False
                })
                full_response = f"{user_response}\n{telemetry_json}"
                return {"response": full_response, "telemetry": telemetry}

        
        # 3. Grounded Prompt Prompt Construction
        context_snippets = []
        logit_bias = {}
        
        # Take top 3 for prompt context (limit 512 tokens roughly)
        used_context_tokens = 0
        for i, (chunk, score) in enumerate(rag_result["results"][:3]):
            snippet = f"[[SOURCE {i}]]: {chunk}"
            context_snippets.append(snippet)
            
            # Extract tokens for logit bias if score is high
            if score > 0.80:
                chunk_tokens = self.tokenizer.encode(chunk)
                for t in chunk_tokens:
                    logit_bias[t] = 0.5 # Small positive bias
            
            used_context_tokens += len(snippet.split()) # Rough count
            if used_context_tokens > 400:
                break
                
        context_block = "\n\n".join(context_snippets)
        
        # STRICT RAG SYSTEM PROMPT
        # Condensed to match training distribution better
        strict_rag_prompt = """
You are MiniGPT. Answer the Q based on the Context.

Rules:
1. Use the Context.
2. Be brief.
"""
        
        final_prompt = (
            f"{strict_rag_prompt}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Q: {query}\n"
            f"A:"
        )
        
        telemetry["grounded"] = True
        telemetry["query_class"] = "GROUNDED"
        
        # 4. Generate
        # Safety thresholds: Stop if avg logprob < -3.0 OR entropy > 1.5
        stopping = {"min_avg_logprob": -3.0, "max_avg_entropy": 1.5}
        
        # Explicitly ban EOS to prevent immediate stop?
        eos_id = getattr(self.tokenizer, 'eos_id', 256)
        # Add strong negative bias to EOS
        if eos_id not in logit_bias:
            logit_bias[eos_id] = -100.0
        else:
            logit_bias[eos_id] -= 100.0
        
        texts, stats_list = self.generate(
            final_prompt,
            max_tokens=64,
            temperature=0.1,  # Strict low temp per instructions
            top_k=40,
            repetition_penalty=1.2,
            logit_bias=logit_bias,
            stopping_criteria=stopping,
            num_return_sequences=1
        )
        response_text = texts[0]
        stats = stats_list[0]
        
        # Check if generation was cut short due to safety
        # (This logic is implicit: if length < max_tokens, it might have stopped)
        avg_logprob = np.mean([s['log_prob'] for s in stats]) if stats else -99
        avg_entropy = np.mean([s['entropy'] for s in stats]) if stats else 99
        telemetry["avg_logprob"] = avg_logprob
        telemetry["avg_entropy"] = avg_entropy
        
        if avg_logprob < -3.0:
             telemetry["fallback_triggered"] = True
             return {"response": "I started answering but became uncertain. Please provide more context.", "telemetry": telemetry}

        return {"response": response_text, "telemetry": telemetry}
