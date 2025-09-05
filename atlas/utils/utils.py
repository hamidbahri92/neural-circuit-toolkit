
import numpy as np
import os, json, hashlib, base64, io
from typing import Tuple

def set_seed(seed: int = 1234):
    np.random.seed(seed)

def save_npy_blob(arr: np.ndarray, store_fn):
    import numpy as np
    bio = io.BytesIO()
    np.save(bio, arr.astype(np.float32))
    ref = store_fn(bio.getvalue())
    return ref

def load_npy_blob(ref: str, load_fn) -> np.ndarray:
    import numpy as np, io
    b = load_fn(ref)
    return np.load(io.BytesIO(b), allow_pickle=False)

def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Solve R = argmin ||RA - B||_F, with R orthogonal. Return R.
    U, _, Vt = np.linalg.svd(B @ A.T, full_matrices=False)
    R = U @ Vt
    return R

def whiten(X: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    # Zero-mean and whiten features in columns. Return (Xw, W) with W s.t. X @ W = Xw
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
    vals, vecs = np.linalg.eigh(C + eps*np.eye(C.shape[0]))
    W = vecs @ np.diag(1.0/np.sqrt(vals)) @ vecs.T
    Xw = Xc @ W
    return Xw, W
