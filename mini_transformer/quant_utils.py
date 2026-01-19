import numpy as np

def quantize_matrix(M):
    """
    Quantizes a matrix M (rows, cols) to INT8 using symmetric per-channel (axis 0) quantization.
    Returns: M_int8, scale (shape (cols,))
    """
    # scale shape: (cols,) - wait, usually per-channel means per-output-channel.
    # If M is (In, Out), we want scale per Out?
    # Or per row?
    # Wx: W is (In, Out). x is (B, In).
    # Matmul: x @ W.
    # Result is (B, Out).
    # Each output column j is dot(x, W[:, j]).
    # We can scale W[:, j] by scale[j].
    # So we want scale per column (axis 0? No, axis 1 is columns).
    # Wait, numpy axis 0 is rows. Axis 1 is cols.
    # If we want scale per column, we take max over axis 0.
    
    max_val = np.max(np.abs(M), axis=0) # (Out,)
    scale = max_val / 127.0
    scale[scale == 0] = 1.0 # Avoid div by zero
    
    # Quantize
    # M / scale -> Broadcast: (Rows, Cols) / (Cols,) -> Works.
    M_int8 = np.round(M / scale).astype(np.int8)
    
    return M_int8, scale.astype(np.float32)

def dequantize_matrix(M_int8, scale):
    """
    Dequantizes M_int8 to float32.
    """
    return M_int8.astype(np.float32) * scale
