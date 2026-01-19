"""
Quantization-Aware Training (QAT) Module

Implements:
- Per-channel INT8 weight quantization
- Asymmetric activation quantization  
- Calibration statistics collection
- QAT fine-tuning pass
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class QuantConfig:
    """Quantization configuration."""
    weight_bits: int = 8
    activation_bits: int = 8
    per_channel: bool = True
    asymmetric: bool = True
    calibration_steps: int = 100


@dataclass 
class QuantStats:
    """Per-layer quantization statistics."""
    min_val: float
    max_val: float
    scale: float
    zero_point: int
    
    def to_dict(self):
        return {
            "min": self.min_val,
            "max": self.max_val, 
            "scale": self.scale,
            "zero_point": self.zero_point
        }


def compute_scale_zp(min_val: float, max_val: float, bits: int = 8, asymmetric: bool = True) -> Tuple[float, int]:
    """
    Compute scale and zero-point for quantization.
    
    For asymmetric quant: x_q = round(x / scale) + zero_point
    For symmetric quant: x_q = round(x / scale), zero_point = 0
    """
    qmin = 0 if asymmetric else -(2 ** (bits - 1))
    qmax = (2 ** bits) - 1 if asymmetric else (2 ** (bits - 1)) - 1
    
    if asymmetric:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))
    else:
        # Symmetric: use max absolute value
        max_abs = max(abs(min_val), abs(max_val))
        scale = max_abs / qmax
        zero_point = 0
    
    return max(scale, 1e-9), zero_point


def quantize_tensor(x: np.ndarray, scale: float, zero_point: int, bits: int = 8) -> np.ndarray:
    """Quantize tensor to INT8."""
    qmin = 0
    qmax = (2 ** bits) - 1
    x_q = np.clip(np.round(x / scale) + zero_point, qmin, qmax).astype(np.int8)
    return x_q


def dequantize_tensor(x_q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize INT8 tensor back to float."""
    return (x_q.astype(np.float32) - zero_point) * scale


def quantize_weights_per_channel(W: np.ndarray, axis: int = 0, bits: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel weight quantization.
    
    Args:
        W: Weight matrix
        axis: Channel axis (0 for output channels, 1 for input channels)
        bits: Number of bits
    
    Returns:
        W_q: Quantized weights (INT8)
        scales: Per-channel scales
        zero_points: Per-channel zero points
    """
    n_channels = W.shape[axis]
    scales = np.zeros(n_channels, dtype=np.float32)
    zero_points = np.zeros(n_channels, dtype=np.int32)
    
    W_q = np.zeros_like(W, dtype=np.int8)
    
    for c in range(n_channels):
        if axis == 0:
            channel_data = W[c]
        else:
            channel_data = W[:, c]
        
        min_val = channel_data.min()
        max_val = channel_data.max()
        
        scale, zp = compute_scale_zp(min_val, max_val, bits, asymmetric=False)
        scales[c] = scale
        zero_points[c] = zp
        
        if axis == 0:
            W_q[c] = quantize_tensor(channel_data, scale, zp, bits)
        else:
            W_q[:, c] = quantize_tensor(channel_data, scale, zp, bits)
    
    return W_q, scales, zero_points


class CalibrationCollector:
    """
    Collects activation statistics during calibration pass.
    Uses running min/max with exponential moving average.
    """
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.stats: Dict[str, QuantStats] = {}
        self.running_min: Dict[str, float] = {}
        self.running_max: Dict[str, float] = {}
    
    def observe(self, name: str, tensor: np.ndarray):
        """Observe tensor values and update running statistics."""
        batch_min = float(tensor.min())
        batch_max = float(tensor.max())
        
        if name not in self.running_min:
            self.running_min[name] = batch_min
            self.running_max[name] = batch_max
        else:
            self.running_min[name] = min(self.running_min[name], batch_min)
            self.running_max[name] = max(self.running_max[name], batch_max)
    
    def finalize(self, bits: int = 8, asymmetric: bool = True):
        """Compute final quantization parameters."""
        for name in self.running_min:
            min_val = self.running_min[name]
            max_val = self.running_max[name]
            scale, zp = compute_scale_zp(min_val, max_val, bits, asymmetric)
            
            self.stats[name] = QuantStats(
                min_val=min_val,
                max_val=max_val,
                scale=scale,
                zero_point=zp
            )
    
    def get_stats(self, name: str) -> Optional[QuantStats]:
        return self.stats.get(name)
    
    def save(self, path: str):
        """Save calibration statistics."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'stats': {k: v.to_dict() for k, v in self.stats.items()},
                'running_min': self.running_min,
                'running_max': self.running_max
            }, f)
        print(f"[QAT] Saved calibration to {path}")
    
    def load(self, path: str):
        """Load calibration statistics."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.running_min = data['running_min']
        self.running_max = data['running_max']
        
        for name, stat_dict in data['stats'].items():
            self.stats[name] = QuantStats(
                min_val=stat_dict['min'],
                max_val=stat_dict['max'],
                scale=stat_dict['scale'],
                zero_point=stat_dict['zero_point']
            )
        print(f"[QAT] Loaded calibration from {path}")


class QATTrainer:
    """
    Quantization-Aware Training orchestrator.
    
    Steps:
    1. Calibration pass (collect activation statistics)
    2. Fake quantization pass (simulate INT8 during training)
    3. Final quantization (export INT8 model)
    """
    
    def __init__(self, model, config: QuantConfig = None):
        self.model = model
        self.config = config or QuantConfig()
        self.collector = CalibrationCollector()
        self.calibrated = False
    
    def calibrate(self, data_generator, n_steps: Optional[int] = None):
        """
        Run calibration pass to collect activation statistics.
        
        Args:
            data_generator: Yields (input_tokens, target_tokens)
            n_steps: Number of calibration steps (overrides config)
        """
        n_steps = n_steps or self.config.calibration_steps
        print(f"[QAT] Starting calibration for {n_steps} steps...")
        
        for step, (inputs, _) in enumerate(data_generator):
            if step >= n_steps:
                break
            
            # Forward pass with observation hooks
            self._calibration_forward(inputs)
            
            if (step + 1) % 20 == 0:
                print(f"[QAT] Calibration step {step + 1}/{n_steps}")
        
        # Finalize statistics
        self.collector.finalize(
            bits=self.config.activation_bits,
            asymmetric=self.config.asymmetric
        )
        self.calibrated = True
        print(f"[QAT] Calibration complete. {len(self.collector.stats)} tensors observed.")
    
    def _calibration_forward(self, inputs):
        """Forward pass with activation observation."""
        if len(inputs.shape) == 1:
            inputs = inputs[np.newaxis, :]
        
        # Get embeddings
        x = self.model.embeddings.forward_seq(inputs[0], 0)
        x = np.array([x])
        self.collector.observe("embeddings_out", x)
        
        # Layers
        for i, layer in enumerate(self.model.layers):
            # Attention
            x_norm = layer.ln1.forward(x)
            self.collector.observe(f"layer_{i}_ln1_out", x_norm)
            
            attn_out, _ = layer.attn.forward(x_norm, 0, None, i)
            self.collector.observe(f"layer_{i}_attn_out", attn_out)
            
            x = x + attn_out
            
            # FFN
            x_norm = layer.ln2.forward(x)
            self.collector.observe(f"layer_{i}_ln2_out", x_norm)
            
            ffn_out = layer.ffn.forward(x_norm)
            self.collector.observe(f"layer_{i}_ffn_out", ffn_out)
            
            x = x + ffn_out
        
        # Final norm
        x = self.model.ln_f.forward(x)
        self.collector.observe("final_norm_out", x)
    
    def quantize_model(self):
        """
        Quantize model weights using collected statistics.
        Returns quantized model info.
        """
        if not self.calibrated:
            raise ValueError("Must calibrate before quantizing")
        
        print("[QAT] Quantizing model weights...")
        
        quant_info = {}
        
        # Quantize embeddings
        W_emb = self.model.embeddings.W_emb
        W_emb_q, scales, zps = quantize_weights_per_channel(W_emb, axis=0)
        quant_info['embeddings'] = {
            'shape': W_emb.shape,
            'scales': scales,
            'compression': f"{W_emb.nbytes / W_emb_q.nbytes:.1f}x"
        }
        
        # Quantize each layer
        for i, layer in enumerate(self.model.layers):
            # Attention W_qkv
            W_qkv = layer.attn.W_qkv
            W_qkv_q, scales, zps = quantize_weights_per_channel(W_qkv, axis=1)
            quant_info[f'layer_{i}_attn_qkv'] = {
                'original_bytes': W_qkv.nbytes,
                'quantized_bytes': W_qkv_q.nbytes + scales.nbytes,
                'compression': f"{W_qkv.nbytes / (W_qkv_q.nbytes + scales.nbytes):.1f}x"
            }
            
            # FFN W1, W2
            W1 = layer.ffn.W1
            W1_q, scales1, _ = quantize_weights_per_channel(W1, axis=1)
            
            W2 = layer.ffn.W2
            W2_q, scales2, _ = quantize_weights_per_channel(W2, axis=1)
            
            quant_info[f'layer_{i}_ffn'] = {
                'W1_compression': f"{W1.nbytes / (W1_q.nbytes + scales1.nbytes):.1f}x",
                'W2_compression': f"{W2.nbytes / (W2_q.nbytes + scales2.nbytes):.1f}x"
            }
        
        print("[QAT] Quantization complete.")
        return quant_info
    
    def save_calibration(self, path: str = "calibration.pkl"):
        self.collector.save(path)
    
    def load_calibration(self, path: str = "calibration.pkl"):
        self.collector.load(path)
        self.calibrated = True


def fake_quantize(x: np.ndarray, scale: float, zero_point: int, bits: int = 8) -> np.ndarray:
    """
    Simulate quantization effects during training (STE - Straight Through Estimator).
    Quantize then dequantize to inject quantization noise.
    """
    x_q = quantize_tensor(x, scale, zero_point, bits)
    x_dq = dequantize_tensor(x_q, scale, zero_point)
    return x_dq


# Demo
def demo():
    print("=== QAT Module Demo ===\n")
    
    # Simulate weight matrix
    W = np.random.randn(256, 512).astype(np.float32)
    print(f"Original weight: {W.shape}, {W.nbytes / 1024:.1f} KB")
    
    # Per-channel quantization
    W_q, scales, zps = quantize_weights_per_channel(W, axis=1)
    print(f"Quantized weight: {W_q.shape}, {W_q.nbytes / 1024:.1f} KB")
    print(f"Scales: {scales.shape}, {scales.nbytes / 1024:.1f} KB")
    print(f"Compression: {W.nbytes / (W_q.nbytes + scales.nbytes):.1f}x")
    
    # Dequantize and check error
    W_dq = np.zeros_like(W)
    for c in range(W.shape[1]):
        W_dq[:, c] = dequantize_tensor(W_q[:, c], scales[c], zps[c])
    
    error = np.abs(W - W_dq).mean()
    print(f"Avg quantization error: {error:.6f}")


if __name__ == "__main__":
    demo()
