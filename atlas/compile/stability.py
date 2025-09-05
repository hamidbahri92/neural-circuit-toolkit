
import numpy as np
from typing import Tuple

def spectral_norm(A: np.ndarray, n_iter: int = 2) -> float:
    x = np.random.randn(A.shape[1]).astype(np.float32)
    x /= (np.linalg.norm(x) + 1e-8)
    for _ in range(n_iter):
        x = A.T @ (A @ x)
        x /= (np.linalg.norm(x) + 1e-8)
    return float(np.linalg.norm(A @ x))

def cap_spectral(W: np.ndarray, dW: np.ndarray, beta: float = 0.9) -> float:
    """Return scale gamma in (0,1] such that ||W + gamma dW||_2 <= beta * ||W||_2."""
    base = spectral_norm(W)
    if base <= 1e-8:
        return 1.0
    gamma = 1.0
    trial = spectral_norm(W + dW)
    if trial <= beta * base:
        return 1.0
    # scale down dW
    gamma = min(1.0, beta * base / (trial + 1e-8))
    return float(max(1e-3, gamma))

def lyapunov_scale(A0_sub: np.ndarray, dA_sub: np.ndarray, margin: float = 0.95) -> float:
    """Return scale gamma so that spectral radius of A0+gamma dA <= margin."""
    A = A0_sub + dA_sub
    eig = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eig)))
    if rho <= margin:
        return 1.0
    return float(max(1e-3, margin / (rho + 1e-8)))
