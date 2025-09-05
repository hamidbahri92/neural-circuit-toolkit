
import numpy as np
from typing import List, Dict, Tuple
from .contrast import balance_weights, confounder_features, mmd2
from .min_pairs import minimal_pairs
from ..models.mock import MockBehaviorModel

def collect_activations(model: MockBehaviorModel, texts: List[str]) -> Dict[str, np.ndarray]:
    out = model.forward(texts)
    return out  # X, Z1, H1, Z2

def differential_salience(model: MockBehaviorModel, behavior: str, pos: List[str], neg: List[str]) -> Dict[str, np.ndarray]:
    # Mean diff in H1 plus gradient weighting (integrated-grad proxy)
    acts_pos = collect_activations(model, pos)
    acts_neg = collect_activations(model, neg)
    dH = acts_pos["H1"].mean(0) - acts_neg["H1"].mean(0)  # d_hidden
    grad = model.grad_wrt_H1(pos+neg, behavior).mean(0)       # d_hidden
    sal = dH * np.abs(grad)                                   # elementwise
    return {"H1": sal}

def iterative_prune_preserve(model: MockBehaviorModel, behavior: str, pos: List[str], neg: List[str], sal_H1: np.ndarray, keep_frac: float = 0.1) -> np.ndarray:
    """Return mask over H1 units that preserves behavior delta while as sparse as possible."""
    n = sal_H1.shape[0]
    order = np.argsort(-np.abs(sal_H1))  # descending
    k = max(1, int(keep_frac * n))
    keep = set(order[:k])
    def behavior_delta(mask):
        # Zero out non-kept units and measure delta in behavior score between pos and neg
        def score(texts):
            out = model.forward(texts)
            H1 = out["H1"].copy()
            nz = np.ones(H1.shape[1], dtype=bool)
            nz[list(set(range(n)) - set(mask))] = False
            H1[:, ~nz] = 0.0
            Z2 = H1 @ model.W2 + model.b2
            idx = {"hedging":0,"formality":1,"refusal":2}[behavior]
            sc = Z2[:, idx]
            return float(sc.mean())
        return score(pos) - score(neg)
    target = behavior_delta(keep)
    # Try to drop lowest contributors while keeping within 80% of target
    for i in range(k, n):
        cand = set(order[:i+1])
        if behavior_delta(cand) >= 0.8 * target:
            keep = cand
        else:
            break
    mask = np.zeros(n, dtype=bool); mask[list(keep)] = True
    return mask

def activation_patching_verify(model: MockBehaviorModel, behavior: str, pos: List[str], neg: List[str], mask: np.ndarray) -> Dict[str, float]:
    # Patch H1 activations from pos into neg on masked units and measure flip
    out_p = model.forward(pos); out_n = model.forward(neg)
    H1_pos = out_p["H1"].mean(0); H1_neg = out_n["H1"].mean(0)
    idx = {"hedging":0,"formality":1,"refusal":2}[behavior]
    # Baselines
    base_pos = float(out_p["Z2"][:,idx].mean()); base_neg = float(out_n["Z2"][:,idx].mean())
    # Patch: take neg H1 and replace masked units by pos means
    H1_patch = H1_neg.copy()
    H1_patch[mask] = H1_pos[mask]
    z2_patch = H1_patch @ model.W2 + model.b2
    patched_neg = float(z2_patch[idx])
    return {"base_pos":base_pos, "base_neg":base_neg, "patched_neg":patched_neg, "delta": patched_neg - base_neg}

def isolation_score(model: MockBehaviorModel, behavior: str, mask: np.ndarray, n_pairs: int = 32) -> float:
    pairs = minimal_pairs(behavior, n_pairs)
    idx = {"hedging":0,"formality":1,"refusal":2}[behavior]
    hits, total = 0, 0
    for a,b in pairs:
        out_a = model.forward([a])["H1"]; out_b = model.forward([b])["H1"]
        # With only masked units, compute behavior score
        def sc(H1):
            H = H1.copy()
            nz = np.zeros(H.shape[1], dtype=bool); nz[mask] = True
            H[:, ~nz] = 0.0
            z2 = H @ model.W2 + model.b2
            return float(z2[:,idx].mean())
        if sc(out_a) > sc(out_b):
            hits += 1
        total += 1
    return hits/float(total)
