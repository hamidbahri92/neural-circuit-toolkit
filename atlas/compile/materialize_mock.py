
import numpy as np
from typing import Dict
from .stability import spectral_norm

def materialize_mask_as_w2_delta(model, mask, head_idx: int, magnitude: float = 0.1) -> np.ndarray:
    """Create a delta for W2[:, head_idx] on masked rows."""
    dW = np.zeros_like(model.W2)
    rows = np.where(mask)[0]
    vec = np.ones_like(rows, dtype=float)
    if len(vec)>0:
        vec = (magnitude / max(1, len(vec))) * vec
        dW[rows, head_idx] += vec
    return dW

def safe_apply_w2_delta(model, dW2: np.ndarray, beta: float = 1.2) -> Dict[str, float]:
    """Scale and apply dW2 to keep spectral norm within beta * base.
    Returns a report with per-application margins.
    """
    base = spectral_norm(model.W2)
    trial = spectral_norm(model.W2 + dW2)
    if trial <= beta * max(1e-6, base):
        gamma = 1.0
    else:
        gamma = (beta * base) / (trial + 1e-8)
    model.W2 = model.W2 + gamma * dW2
    final = spectral_norm(model.W2)
    util = final / (beta * max(base, 1e-6))  # utilization of spectral budget
    return {"gamma": float(gamma), "base": float(base), "trial": float(trial), "final": float(final), "utilization": float(util)}
