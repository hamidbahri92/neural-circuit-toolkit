
import numpy as np
from typing import Dict, Any, Tuple
from ..utils.utils import orthogonal_procrustes, whiten

def procrustes_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A and B: k x d bases (rows are basis vectors). Return rotation R such that R@A ≈ B.
    return orthogonal_procrustes(A, B)

def dynamics_fingerprint(acts: np.ndarray) -> Dict[str, Any]:
    # acts: N x d activations for a layer. Return simple dynamics proxy.
    # Sensitivity spectrum: top-k singular values of grad surrogate ~ covariance
    C = (acts.T @ acts) / max(1, acts.shape[0]-1)
    svals = np.linalg.svd(C, compute_uv=False)[:16]
    # Saturation proxy: percentiles
    pct = np.percentile(acts, [5,25,50,75,95], axis=0).mean(axis=-1)
    return {"spectrum": svals.astype(float).tolist(), "sat": pct.astype(float).tolist()}

def whiten_then_procrustes(X_src: np.ndarray, X_tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, Ws = whiten(X_src)
    Xt, Wt = whiten(X_tgt)
    # Compute bases (top-k right singular vectors)
    ks = min(32, Xs.shape[1], Xs.shape[0])
    kt = min(32, Xt.shape[1], Xt.shape[0])
    U_s, _, _ = np.linalg.svd(Xs, full_matrices=False)
    U_t, _, _ = np.linalg.svd(Xt, full_matrices=False)
    k = min(ks, kt)
    R = procrustes_align(U_s[:,:k].T, U_t[:,:k].T)
    return R, Ws, Wt
