"""
Robot-Safe Inference Kernel - Separates thinking from acting.

This module wraps the transformer and meta-controller to provide
a robot-safe inference interface. Key principle: THINK BEFORE ACTING.

The kernel:
1. Runs transformer inference (thinking)
2. Evaluates cognitive state via MetaController
3. Returns intention, NOT action
4. Action execution is a separate, gated step
"""

import numpy as np
import sys
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass

sys.path.append(os.getcwd())

from mini_transformer.transformer import MiniTransformer
from mini_transformer.cognitive import MetaController, CognitiveState, ActionGate


@dataclass
class Intention:
    """
    Represents the model's intention BEFORE action execution.
    Robot controller should inspect this before moving.
    """
    token_id: int
    token_str: str
    logprob: float
    cognitive_state: CognitiveState
    
    # Derived flags for easy checking
    should_act: bool = False
    should_hesitate: bool = False
    should_clarify: bool = False
    should_explore: bool = False
    should_stop: bool = False
    
    def __post_init__(self):
        gate = self.cognitive_state.action_gate
        self.should_act = gate == ActionGate.ACT
        self.should_hesitate = gate == ActionGate.HESITATE
        self.should_clarify = gate == ActionGate.CLARIFY
        self.should_explore = gate == ActionGate.EXPLORE
        self.should_stop = gate == ActionGate.STOP


class RobotInferenceKernel:
    """
    Robot-safe inference wrapper that separates thinking from acting.
    
    Usage:
        kernel = RobotInferenceKernel(model, tokenizer)
        
        # Step 1: Think (inference)
        intention = kernel.think(input_tokens)
        
        # Step 2: Check cognitive state
        if intention.should_act:
            # Step 3: Execute action (motor output)
            action = kernel.act(intention)
            robot.execute(action)
        elif intention.should_clarify:
            robot.speak("Can you clarify?")
        elif intention.should_hesitate:
            robot.pause(0.5)  # Brief pause
    """
    
    def __init__(self, model: MiniTransformer, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.meta_controller = MetaController()
        
        # Inference state
        self.current_tokens: List[int] = []
        self.position = 0
        
        # Logging
        self.intention_log: List[Intention] = []
        self.debug_mode = True
        
    def reset(self):
        """Reset for new conversation."""
        self.current_tokens = []
        self.position = 0
        self.meta_controller.reset()
        self.intention_log = []
        
        # Reset KV cache if model has it
        if hasattr(self.model, 'kv_cache') and hasattr(self.model.kv_cache, 'reset'):
            self.model.kv_cache.reset()
    
    def think(self, 
              input_tokens: Optional[List[int]] = None,
              temperature: float = 0.9,
              top_k: int = 40) -> Intention:
        """
        Run inference and return intention WITHOUT executing action.
        
        Args:
            input_tokens: New tokens to process (prompt or continuation)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        
        Returns:
            Intention object with cognitive state and action gate
        """
        # Add new tokens if provided
        if input_tokens:
            if self.position == 0:
                # Initial prompt
                self.current_tokens = list(input_tokens)
            else:
                # Continuation
                self.current_tokens.extend(input_tokens)
        
        # Forward pass
        tokens_to_process = self.current_tokens[self.position:]
        if not tokens_to_process:
            # Need at least one token
            raise ValueError("No tokens to process")
        
        logits, attn_weights = self.model.forward(
            np.array(tokens_to_process), 
            start_pos=self.position
        )
        
        # Sample next token
        token_logits = logits[0, -1].copy()
        
        # Temperature
        token_logits = token_logits / (temperature + 1e-9)
        
        # Top-k
        if top_k > 0:
            kth_val = np.partition(token_logits, -top_k)[-top_k]
            token_logits[token_logits < kth_val] = -float('inf')
        
        # Softmax
        max_logit = np.max(token_logits)
        if max_logit == -float('inf'):
            probs = np.ones_like(token_logits) / len(token_logits)
        else:
            exp_logits = np.exp(token_logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
        
        # Sample
        next_token = np.random.choice(len(probs), p=probs)
        logprob = np.log(probs[next_token] + 1e-9)
        
        # Memory utilization
        memory_util = 0.0
        if hasattr(self.model, 'kv_cache'):
            cache = self.model.kv_cache
            memory_util = cache.current_len / cache.max_len
        
        # Cognitive evaluation
        cognitive_state = self.meta_controller.step(
            attn_weights=attn_weights,
            logprob=logprob,
            chosen_token=next_token,
            memory_utilization=memory_util
        )
        
        # Create intention
        intention = Intention(
            token_id=next_token,
            token_str=self.tokenizer.decode([next_token]),
            logprob=logprob,
            cognitive_state=cognitive_state
        )
        
        # Log
        self.intention_log.append(intention)
        
        if self.debug_mode:
            self._log_intention(intention)
        
        return intention
    
    def act(self, intention: Intention) -> Optional[str]:
        """
        Execute the intended action (commit token to sequence).
        
        Only call this if intention.should_act is True.
        Returns the generated token string.
        """
        if not intention.should_act:
            if self.debug_mode:
                print(f"[Robot] Action blocked: {intention.cognitive_state.gate_reason}")
            return None
        
        # Commit token to sequence
        self.current_tokens.append(intention.token_id)
        self.position = len(self.current_tokens)
        
        return intention.token_str
    
    def force_act(self, intention: Intention) -> str:
        """
        Force action execution regardless of cognitive state.
        Use with caution - bypasses safety checks.
        """
        self.current_tokens.append(intention.token_id)
        self.position = len(self.current_tokens)
        return intention.token_str
    
    def generate_with_gating(self, 
                              prompt: str, 
                              max_tokens: int = 50,
                              hesitate_callback=None,
                              clarify_callback=None,
                              explore_callback=None) -> str:
        """
        Full generation loop with cognitive gating.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            hesitate_callback: Called when hesitating (optional)
            clarify_callback: Called when needing clarification (optional)
            explore_callback: Called when exploring (optional)
        
        Returns:
            Generated text
        """
        self.reset()
        
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        
        generated_tokens = []
        stop_reason = None
        
        # Use think() for initial prefill by passing prompt tokens
        self.current_tokens = []
        self.position = 0
        
        for i in range(max_tokens):
            # For first iteration, pass prompt tokens
            if i == 0:
                intention = self.think(input_tokens=prompt_tokens)
            else:
                # Add previous token to context and think about next
                self.current_tokens.append(intention.token_id)
                intention = self.think(input_tokens=[intention.token_id])
            
            # Gate actions
            if intention.should_stop:
                stop_reason = "stop_gate"
                break
            elif intention.should_clarify:
                if clarify_callback:
                    clarify_callback(intention)
                stop_reason = "clarify_needed"
                break
            elif intention.should_explore:
                if explore_callback:
                    explore_callback(intention)
                # Continue anyway but with exploration flag
            elif intention.should_hesitate:
                if hesitate_callback:
                    hesitate_callback(intention)
                # Continue but log
            
            # Print character
            sys.stdout.write(intention.token_str)
            sys.stdout.flush()
            generated_tokens.append(intention.token_str)
        
        # Final token commit
        if intention:
            self.current_tokens.append(intention.token_id)
        
        if stop_reason:
            print(f"\n[Robot] Generation stopped: {stop_reason}")
        
        return self.tokenizer.decode(self.current_tokens)
    
    def get_cognitive_summary(self) -> dict:
        """
        Get summary of cognitive state over the generation.
        """
        if not self.intention_log:
            return {}
        
        entropies = [i.cognitive_state.entropy for i in self.intention_log]
        confidences = [i.cognitive_state.confidence for i in self.intention_log]
        
        gate_counts = {}
        for intention in self.intention_log:
            gate = intention.cognitive_state.action_gate.value
            gate_counts[gate] = gate_counts.get(gate, 0) + 1
        
        return {
            "total_steps": len(self.intention_log),
            "avg_entropy": np.mean(entropies),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "gate_counts": gate_counts,
            "head_analysis": self.meta_controller.analyze_heads(
                self.model.forward(np.array([self.current_tokens[-1]]), 
                                  start_pos=len(self.current_tokens)-1)[1]
            ) if self.current_tokens else {}
        }
    
    def _log_intention(self, intention: Intention):
        """Debug logging for intention."""
        state = intention.cognitive_state
        gate = state.action_gate.value.upper()
        
        # Compact log format
        log_line = (f"[Cog] E:{state.entropy:.2f} C:{state.confidence:.2f} "
                   f"M:{state.memory_pressure:.1%} A:{state.head_agreement:.2f} "
                   f"→ {gate}")
        
        if state.gate_reason != "high_confidence":
            log_line += f" ({state.gate_reason})"
        
        print(log_line)


def demo():
    """Demo the robot inference kernel."""
    import pickle
    from mini_transformer.tokenizer import BPETokenizer, TokenizerConfig
    
    print("=== Robot Inference Kernel Demo ===\n")
    
    # Load tokenizer
    if not os.path.exists("models/tokenizer.model"):
        print("[Error] models/tokenizer.model not found. Run train.py first.")
        return
    
    config = TokenizerConfig(vocab_size=2048)
    tokenizer = BPETokenizer(config)
    tokenizer.load("models/tokenizer.model")
    
    # Load model
    model_path = "models/mini_transformer_model.pkl"
    if not os.path.exists(model_path):
        print("[Error] Model not found. Run train.py first.")
        return
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Reset KV cache
    from mini_transformer.kv_cache import KVCache
    model.kv_cache = KVCache(256, model.n_kv_heads, model.d_model // model.n_heads, model.n_layers)
    
    # Create kernel
    kernel = RobotInferenceKernel(model, tokenizer)
    kernel.debug_mode = True
    
    # Callbacks
    def on_hesitate(intention):
        print(f"\n[Robot] Hesitating... (reason: {intention.cognitive_state.gate_reason})")
    
    def on_clarify(intention):
        print(f"\n[Robot] Need clarification! Entropy: {intention.cognitive_state.entropy:.2f}")
    
    def on_explore(intention):
        print(f"\n[Robot] Exploring alternative... (loop detected)")
    
    # Generate
    print("\n--- Generation with Cognitive Gating ---\n")
    result = kernel.generate_with_gating(
        prompt="Hello Robot",
        max_tokens=30,
        hesitate_callback=on_hesitate,
        clarify_callback=on_clarify,
        explore_callback=on_explore
    )
    
    print(f"\n\n--- Final Output ---\n'{result}'")
    
    # Summary
    print("\n--- Cognitive Summary ---")
    summary = kernel.get_cognitive_summary()
    for k, v in summary.items():
        if k != "head_analysis":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    demo()
