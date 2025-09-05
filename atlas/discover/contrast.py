
import numpy as np
from typing import List, Tuple, Dict
from ..semantics.encoder import ByteNGramEncoder

def confounder_features(texts: List[str], encoder: ByteNGramEncoder) -> np.ndarray:
    # Simple confounders: length, punctuation ratio, semantic embedding (low-dim proj)
    lens = np.array([len(t) for t in texts], dtype=np.float32)[:,None]
    punct = np.array([sum(ch in ',.;:!?-' for ch in t)/max(1,len(t)) for t in texts], dtype=np.float32)[:,None]
    emb = np.stack([encoder.encode(t) for t in texts], axis=0).astype(np.float32)
    # reduce embedding by random projection to 16 dims for stability
    rng = np.random.RandomState(1234)
    P = rng.normal(size=(emb.shape[1], 16)).astype(np.float32) / np.sqrt(emb.shape[1])
    proj = emb @ P
    X = np.concatenate([lens, punct, proj], axis=1)
    # standardize
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    return X

def balance_weights(pos_texts: List[str], neg_texts: List[str], encoder: ByteNGramEncoder, l2: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nonnegative importance weights to match confounder moments.
    We solve a ridge-regularized least squares to make the (weighted) pos mean match neg mean.
    """
    Xp = confounder_features(pos_texts, encoder)   # n_p x d
    Xn = confounder_features(neg_texts, encoder)   # n_n x d
    mu_n = Xn.mean(0)                               # d
    # Solve min_w ||Xp^T w / sum(w) - mu_n||^2 + l2||w-1||^2, w>=0
    # Approximate by ignoring normalization in denominator first, then renormalize.
    A = Xp
    b = mu_n * len(pos_texts)
    AtA = A.T @ A + l2 * np.eye(A.shape[1], dtype=np.float32)
    w_lin = A @ np.linalg.solve(AtA, b)
    w = np.maximum(w_lin, 1e-3).astype(np.float32)
    # Normalize to have mean weight ~ 1
    w *= len(pos_texts) / (w.sum() + 1e-8)
    # For negatives, set uniform weights
    w_neg = np.ones(len(neg_texts), dtype=np.float32)
    return w, w_neg

def mmd2(Xw: np.ndarray, Yw: np.ndarray, sigma: float = 1.0) -> float:
    """Squared Maximum Mean Discrepancy with RBF kernel, using weights as sample weights.
    Xw, Yw: tuples (X, w) where X is nxd, w is n
    """
    X, wx = Xw; Y, wy = Yw
    def k(a,b):
        aa = (a*a).sum(1, keepdims=True)
        bb = (b*b).sum(1, keepdims=True).T
        D = aa + bb - 2*a@b.T
        return np.exp(-D/(2*sigma**2))
    Kxx = k(X,X); Kyy = k(Y,Y); Kxy = k(X,Y)
    wxn = wx/(wx.sum()+1e-8); wyn = wy/(wy.sum()+1e-8)
    mmd = (wxn[:,None]*wxn[None,:]*Kxx).sum() + (wyn[:,None]*wyn[None,:]*Kyy).sum() - 2*(wxn[:,None]*wyn[None,:]*Kxy).sum()
    return float(mmd)
