
import numpy as np
from typing import Dict, List, Tuple
from ..semantics.encoder import ByteNGramEncoder

class MockBehaviorModel:
    """A tiny 2-layer MLP over byte-ngrams that simulates three behaviors.
    Exposes activations and simple gradients for discovery experiments.
    """
    def __init__(self, encoder: ByteNGramEncoder, d_hidden: int = 64, seed: int = 42):
        self.encoder = encoder
        rng = np.random.RandomState(seed)
        d_in = encoder.dim
        self.W1 = rng.normal(scale=0.2, size=(d_in, d_hidden)).astype(np.float32)
        self.b1 = np.zeros((d_hidden,), dtype=np.float32)
        self.W2 = rng.normal(scale=0.2, size=(d_hidden, 3)).astype(np.float32) # 3 behaviors
        self.b2 = np.zeros((3,), dtype=np.float32)
        # Bake in behavior-specific directions by adding known ngram vectors
        def bump(phrase, head_idx, scale=1.0):
            v = self.encoder.encode(phrase)
            self.W1 += np.outer(v, (scale*rng.normal(size=(d_hidden,))).astype(np.float32)) * 1e-3
            self.W2[:, head_idx] += (scale * rng.normal(size=(d_hidden,))).astype(np.float32) * 1e-2
        for p in ["maybe","perhaps","it might"]: bump(p, 0, 2.0)  # hedging
        for p in ["therefore","moreover","thus"]: bump(p, 1, 2.0) # formality
        for p in ["I cannot","won't do","inappropriate"]: bump(p, 2, 2.0) # refusal

    def forward(self, texts: List[str]) -> Dict[str, np.ndarray]:
        X = np.stack([self.encoder.encode(t) for t in texts], axis=0)  # n x d
        Z1 = X @ self.W1 + self.b1
        H1 = np.maximum(Z1, 0.0)  # ReLU
        Z2 = H1 @ self.W2 + self.b2  # n x 3
        return {"X":X, "Z1":Z1, "H1":H1, "Z2":Z2}

    def behavior_scores(self, texts: List[str]) -> Dict[str, np.ndarray]:
        Z2 = self.forward(texts)["Z2"]
        return {"hedging":Z2[:,0], "formality":Z2[:,1], "refusal":Z2[:,2]}

    def grad_wrt_H1(self, texts: List[str], behavior: str) -> np.ndarray:
        """Gradient of behavior score wrt H1 (pre-W2 activations)."""
        idx = {"hedging":0,"formality":1,"refusal":2}[behavior]
        # For linear head, grad wrt H1 is just W2[:, idx]
        return np.tile(self.W2[:, idx][None, :], (len(texts),1)).astype(np.float32)
