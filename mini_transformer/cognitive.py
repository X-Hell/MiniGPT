"""
Cognitive Robotics Controller - Meta-cognitive layer for robot-safe inference.

This module provides monitoring and gating of transformer inference for
embodied AI applications. It tracks internal states (entropy, confidence,
memory pressure) and makes decisions about when to ACT, HESITATE, or CLARIFY.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class ActionGate(Enum):
    """Possible action decisions from the meta-controller."""
    ACT = "act"           # Confident, execute action
    HESITATE = "hesitate" # Uncertain, pause and observe
    CLARIFY = "clarify"   # Low confidence, request clarification
    STOP = "stop"         # Entropy too high, abort
    EXPLORE = "explore"   # Loop detected, try something new


@dataclass
class CognitiveState:
    """
    Central state struct tracking all cognitive metrics.
    Updated after each forward pass by the MetaController.
    """
    # Attention metrics
    entropy: float = 0.0              # Avg attention entropy across heads
    head_agreement: float = 1.0       # Min pairwise cosine similarity of heads
    head_entropies: List[float] = field(default_factory=list)
    
    # Confidence metrics
    confidence: float = 0.0           # Log probability of chosen token
    confidence_streak: int = 0        # Consecutive high/low confidence tokens
    
    # Memory metrics
    memory_pressure: float = 0.0      # KV cache utilization (0-1)
    
    # Repetition metrics
    repetition_score: float = 0.0     # N-gram overlap with recent tokens
    loop_detected: bool = False
    
    # Decision
    action_gate: ActionGate = ActionGate.ACT
    gate_reason: str = ""
    
    def to_dict(self):
        return {
            "entropy": round(self.entropy, 3),
            "head_agreement": round(self.head_agreement, 3),
            "confidence": round(self.confidence, 3),
            "memory_pressure": round(self.memory_pressure, 3),
            "repetition_score": round(self.repetition_score, 3),
            "action_gate": self.action_gate.value,
            "gate_reason": self.gate_reason
        }


class EntropyGate:
    """
    Monitors attention entropy and gates actions when uncertainty is high.
    
    Thresholds:
    - entropy < 2.0: Confident, proceed
    - entropy ∈ [2.0, 2.5]: Moderate uncertainty, cautious
    - entropy ∈ [2.5, 3.5]: High uncertainty, hesitate
    - entropy > 3.5: Very uncertain, stop
    """
    
    def __init__(self, hesitate_threshold=2.5, stop_threshold=3.5):
        self.hesitate_threshold = hesitate_threshold
        self.stop_threshold = stop_threshold
    
    def evaluate(self, attn_weights: np.ndarray) -> Tuple[float, Optional[ActionGate]]:
        """
        Compute entropy from attention weights.
        attn_weights: (1, n_heads, T_q, T_k)
        Returns: (entropy, suggested_gate or None)
        """
        # Compute entropy: H = -sum(p * log(p))
        p = attn_weights[0]  # (n_heads, T_q, T_k)
        log_p = np.log(p + 1e-9)
        entropy_per_head = -np.sum(p * log_p, axis=-1)  # (n_heads, T_q)
        avg_entropy = entropy_per_head.mean()
        
        if avg_entropy > self.stop_threshold:
            return avg_entropy, ActionGate.STOP
        elif avg_entropy > self.hesitate_threshold:
            return avg_entropy, ActionGate.HESITATE
        
        return avg_entropy, None


class HeadDisagreementDetector:
    """
    Detects when attention heads disagree, suggesting ambiguity.
    Requires consensus before executing actions.
    
    Note: Small models naturally have lower head agreement.
    Threshold should be tuned per model size.
    """
    
    def __init__(self, min_agreement=-0.3):
        self.min_agreement = min_agreement
    
    def evaluate(self, attn_weights: np.ndarray) -> Tuple[float, bool]:
        """
        Compute pairwise cosine similarity of head attention patterns.
        attn_weights: (1, n_heads, T_q, T_k)
        Returns: (min_agreement, heads_disagree)
        """
        n_heads = attn_weights.shape[1]
        
        # Flatten attention patterns per head
        patterns = attn_weights[0].reshape(n_heads, -1)  # (n_heads, T_q * T_k)
        
        # Normalize
        norms = np.linalg.norm(patterns, axis=1, keepdims=True) + 1e-9
        patterns_norm = patterns / norms
        
        # Compute pairwise cosine similarity
        sim_matrix = np.matmul(patterns_norm, patterns_norm.T)
        
        # Get minimum off-diagonal similarity
        mask = ~np.eye(n_heads, dtype=bool)
        min_sim = sim_matrix[mask].min()
        
        return min_sim, min_sim < self.min_agreement


class ConfidenceActor:
    """
    Decides actions based on token log probability (confidence).
    
    Thresholds:
    - logprob > -1.5: High confidence → ACT
    - logprob ∈ [-3.0, -1.5]: Medium confidence → HESITATE
    - logprob < -3.0: Low confidence → CLARIFY
    """
    
    def __init__(self, high_conf=-1.5, low_conf=-3.0):
        self.high_conf = high_conf
        self.low_conf = low_conf
    
    def evaluate(self, logprob: float) -> Tuple[ActionGate, str]:
        """
        Returns action gate based on confidence.
        """
        if logprob > self.high_conf:
            return ActionGate.ACT, "high_confidence"
        elif logprob > self.low_conf:
            return ActionGate.HESITATE, "medium_confidence"
        else:
            return ActionGate.CLARIFY, "low_confidence"


class LoopBreaker:
    """
    Detects repetitive output patterns (thought loops) and triggers exploration.
    Uses N-gram overlap detection.
    """
    
    def __init__(self, ngram_size=3, overlap_threshold=0.5, window_size=20):
        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold
        self.window_size = window_size
    
    def evaluate(self, token_history: List[int]) -> Tuple[float, bool]:
        """
        Compute N-gram repetition score.
        Returns: (repetition_score, loop_detected)
        """
        if len(token_history) < self.ngram_size * 2:
            return 0.0, False
        
        # Get recent tokens
        recent = token_history[-self.window_size:]
        
        # Extract N-grams
        ngrams = []
        for i in range(len(recent) - self.ngram_size + 1):
            ngrams.append(tuple(recent[i:i + self.ngram_size]))
        
        if not ngrams:
            return 0.0, False
        
        # Count duplicates
        unique_ngrams = set(ngrams)
        repetition_score = 1.0 - (len(unique_ngrams) / len(ngrams))
        
        return repetition_score, repetition_score > self.overlap_threshold


class MemoryPressureMonitor:
    """
    Monitors KV cache utilization and suggests simplification when memory is high.
    """
    
    def __init__(self, high_pressure=0.8, critical_pressure=0.95):
        self.high_pressure = high_pressure
        self.critical_pressure = critical_pressure
    
    def evaluate(self, utilization: float) -> Tuple[str, Optional[ActionGate]]:
        """
        Returns memory status and suggested gate.
        """
        if utilization > self.critical_pressure:
            return "critical", ActionGate.STOP
        elif utilization > self.high_pressure:
            return "high", ActionGate.HESITATE
        return "normal", None


# Head role definitions
HEAD_ROLES = {
    0: "syntax",   # Grammar, word order patterns
    1: "memory",   # Long-range dependencies
    2: "intent",   # Action verbs, goals
    3: "context",  # Local adjacent tokens
}


class MetaController:
    """
    Orchestrates all cognitive monitors and produces a unified CognitiveState.
    This is the "prefrontal cortex" of the inference engine.
    """
    
    def __init__(self):
        self.entropy_gate = EntropyGate()
        self.disagreement_detector = HeadDisagreementDetector()
        self.confidence_actor = ConfidenceActor()
        self.loop_breaker = LoopBreaker()
        self.memory_monitor = MemoryPressureMonitor()
        
        # History for loop detection
        self.token_history: List[int] = []
        
        # Decision log for debugging
        self.decision_log: List[dict] = []
    
    def step(self, 
             attn_weights: np.ndarray, 
             logprob: float,
             chosen_token: int,
             memory_utilization: float = 0.0) -> CognitiveState:
        """
        Process one inference step and return CognitiveState with action decision.
        
        Args:
            attn_weights: (1, n_heads, T_q, T_k) attention patterns
            logprob: Log probability of chosen token
            chosen_token: Token ID that was sampled
            memory_utilization: KV cache utilization (0-1)
        
        Returns:
            CognitiveState with all metrics and action_gate decision
        """
        state = CognitiveState()
        
        # Update token history
        self.token_history.append(chosen_token)
        if len(self.token_history) > 100:  # Keep last 100
            self.token_history = self.token_history[-100:]
        
        # 1. Entropy evaluation
        entropy, entropy_gate = self.entropy_gate.evaluate(attn_weights)
        state.entropy = entropy
        
        # Per-head entropy
        p = attn_weights[0]
        log_p = np.log(p + 1e-9)
        state.head_entropies = (-np.sum(p * log_p, axis=-1).mean(axis=-1)).tolist()
        
        # 2. Head disagreement
        agreement, heads_disagree = self.disagreement_detector.evaluate(attn_weights)
        state.head_agreement = agreement
        
        # 3. Confidence
        state.confidence = logprob
        conf_gate, conf_reason = self.confidence_actor.evaluate(logprob)
        
        # 4. Loop detection
        rep_score, loop_detected = self.loop_breaker.evaluate(self.token_history)
        state.repetition_score = rep_score
        state.loop_detected = loop_detected
        
        # 5. Memory pressure
        state.memory_pressure = memory_utilization
        mem_status, mem_gate = self.memory_monitor.evaluate(memory_utilization)
        
        # === Decision Logic (Priority Order) ===
        # 1. Memory critical → STOP
        # 2. Entropy critical → STOP
        # 3. Loop detected → EXPLORE
        # 4. Heads disagree → HESITATE
        # 5. Entropy high → HESITATE
        # 6. Low confidence → CLARIFY
        # 7. Otherwise → use confidence actor
        
        if mem_gate == ActionGate.STOP:
            state.action_gate = ActionGate.STOP
            state.gate_reason = "memory_critical"
        elif entropy_gate == ActionGate.STOP:
            state.action_gate = ActionGate.STOP
            state.gate_reason = "entropy_critical"
        elif loop_detected:
            state.action_gate = ActionGate.EXPLORE
            state.gate_reason = "loop_detected"
        elif heads_disagree:
            state.action_gate = ActionGate.HESITATE
            state.gate_reason = "head_disagreement"
        elif entropy_gate == ActionGate.HESITATE:
            state.action_gate = ActionGate.HESITATE
            state.gate_reason = "entropy_high"
        elif mem_gate == ActionGate.HESITATE:
            state.action_gate = ActionGate.HESITATE
            state.gate_reason = "memory_high"
        else:
            state.action_gate = conf_gate
            state.gate_reason = conf_reason
        
        # Log decision
        self.decision_log.append(state.to_dict())
        
        return state
    
    def reset(self):
        """Reset state for new conversation."""
        self.token_history = []
        self.decision_log = []
    
    def get_head_role(self, head_idx: int) -> str:
        """Get the functional role of a head."""
        return HEAD_ROLES.get(head_idx, "unknown")
    
    def analyze_heads(self, attn_weights: np.ndarray) -> dict:
        """
        Analyze which heads are most active for current token.
        Returns per-head analysis.
        """
        n_heads = attn_weights.shape[1]
        p = attn_weights[0]  # (n_heads, T_q, T_k)
        
        analysis = {}
        for h in range(n_heads):
            role = self.get_head_role(h)
            entropy = -np.sum(p[h] * np.log(p[h] + 1e-9))
            max_attn = p[h].max()
            analysis[f"head_{h}_{role}"] = {
                "entropy": float(entropy),
                "max_attention": float(max_attn),
                "focused": entropy < 1.5
            }
        
        return analysis
