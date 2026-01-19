import numpy as np

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs)  # (end, dim//2)
    freqs_cis = np.exp(1j * freqs)  # complex64
    return freqs_cis

def apply_rope(xq, xk, freqs_cis):
    """
    xq: (B, H, T, D)
    xk: (B, H, T, D)
    freqs_cis: (T, D//2) - or broadcastable
    """
    # Ensure contiguous and float32
    xq = np.ascontiguousarray(xq).astype(np.float32)
    xk = np.ascontiguousarray(xk).astype(np.float32)
    
    # Reshape to complex
    xq_ = xq.view(np.complex64)
    xk_ = xk.view(np.complex64)
    
    # Broadcast freqs_cis
    # freqs_cis needs to align with T dim.
    # xq shape: (B, H, T, D_head/2 complex)
    # freqs_cis shape: (T, D_head/2) -> reshape to (1, 1, T, D_head/2)
    freqs_cis = freqs_cis[None, None, :, :]
    
    # Rotate
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis
    
    # Flatten back to float
    # We must assume the output should be flattened such that real/imag are interleaved or concatenated?
    # view(np.float32) on complex array of shape (..., D/2) results in (..., D/2, 2).
    # We typically want (..., D).
    # So we simply flatten the last dimension.
    
    return xq_out.view(np.float32).reshape(*xq.shape), xk_out.view(np.float32).reshape(*xk.shape)

def apply_rope_backward(dxq, dxk, freqs_cis):
    """
    Backward pass for RoPE is rotation by negative angle (conjugate).
    """
    freqs_cis_conj = np.conj(freqs_cis)
    return apply_rope(dxq, dxk, freqs_cis_conj)
