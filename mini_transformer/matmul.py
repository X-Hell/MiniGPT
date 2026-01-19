import numpy as np
import time

class MatMulLogger:
    """Logs memory and shape info for explicit matrix multiplications."""
    def __init__(self):
        self.log = []
        self.total_macs = 0

    def log_matmul(self, name, A, B, C):
        """
        A: (M, K)
        B: (K, N)
        C: (M, N)
        """
        assert A.shape[-1] == B.shape[-2], f"Shape mismatch: {A.shape} vs {B.shape}"
        
        mem_a = A.nbytes
        mem_b = B.nbytes
        mem_c = C.nbytes
        total_mem = mem_a + mem_b + mem_c
        
        # Flops approx 2*M*N*K
        macs = np.prod(C.shape) * A.shape[-1]
        self.total_macs += macs

        entry = {
            "name": name,
            "A_shape": A.shape,
            "B_shape": B.shape,
            "C_shape": C.shape,
            "mem_bytes": total_mem, # Instantaneous memory for this op
        }
        self.log.append(entry)
        
        # Print explicit info as requested
        print(f"[MatMul] {name:20s} | {str(A.shape):>12s} x {str(B.shape):>12s} -> {str(C.shape):>12s} | Mem: {total_mem/1024:.2f} KB")

_LOGGER = MatMulLogger()

def explicit_matmul(A, B, name="matmul"):
    """
    Wrapper for np.matmul that logs shapes and memory usage.
    """
    start = time.time()
    C = np.matmul(A, B)
    _LOGGER.log_matmul(name, A, B, C)
    return C

def get_stats():
    return _LOGGER.log
