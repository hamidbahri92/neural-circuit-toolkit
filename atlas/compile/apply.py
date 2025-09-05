
import numpy as np
from typing import Dict, Any, Tuple

def apply_lowrank(W: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    # Returns W + U @ V^T
    return W + U @ V.T

def apply_row_edit(W: np.ndarray, rows: np.ndarray, delta: np.ndarray) -> np.ndarray:
    W2 = W.copy()
    W2[rows] += delta
    return W2

def spectral_norm(A: np.ndarray, n_iter: int = 2) -> float:
    x = np.random.randn(A.shape[1])
    for _ in range(n_iter):
        x = A.T @ (A @ x)
        n = np.linalg.norm(x)+1e-8
        x /= n
    return float(np.linalg.norm(A @ x))
