import numpy as np
import time

class MatMulLogger:
    """Logs memory and shape info for explicit matrix multiplications."""
    def __init__(self):
        self.log = []
        self.total_macs = 0
        self.mem_peak = 0

    def log_matmul(self, name, A, B, C):
        """
        A: (M, K)
        B: (K, N)
        C: (M, N)
        """
        if hasattr(A, 'nbytes'):
             mem_a = A.nbytes
             mem_b = B.nbytes
             mem_c = C.nbytes
             total_mem = mem_a + mem_b + mem_c
        else:
             total_mem = 0
             
        # Peak Tracking
        if total_mem > self.mem_peak:
            self.mem_peak = total_mem
        
        # Flops approx 2*M*N*K
        macs = np.prod(C.shape) * A.shape[-1]
        self.total_macs += macs

        entry = {
            "name": name,
            "A_shape": A.shape,
            "B_shape": B.shape,
            "C_shape": C.shape,
            "mem_bytes": total_mem, 
        }
        self.log.append(entry)
        
        # Print explicit info as requested
        print(f"[MatMul] {name:20s} | {str(A.shape):>12s} x {str(B.shape):>12s} -> {str(C.shape):>12s} | Mem: {total_mem/1024:.2f} KB")

_LOGGER = MatMulLogger()

class BufferManager:
    """
    Manages a single large scratchpad buffer to reuse memory.
    """
    def __init__(self, size_bytes):
        self.size_bytes = size_bytes
        self.buffer = np.zeros(size_bytes // 4, dtype=np.float32)
        print(f"[BufferManager] Allocated scratchpad: {size_bytes/1024/1024:.2f} MB")
        
    def get(self, shape):
        """Returns a view of the buffer with desired shape."""
        req_elements = int(np.prod(shape))
        if req_elements > self.buffer.size:
             print(f"[Buffer] Resize needed! {self.buffer.size} -> {req_elements}")
             self.buffer = np.zeros(req_elements, dtype=np.float32)
             
        return self.buffer[:req_elements].reshape(shape)

def explicit_matmul(A, B, name="matmul", out=None):
    """
    Wrapper for np.matmul that logs shapes and memory usage.
    Supports optional 'out' buffer for memory reuse.
    """
    if out is not None:
        np.matmul(A, B, out=out)
        C = out
    else:
        C = np.matmul(A, B)
        
    _LOGGER.log_matmul(name, A, B, C)
    return C

def get_stats():
    return _LOGGER.log
