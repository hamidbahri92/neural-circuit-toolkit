
import numpy as np, hashlib
from typing import List

class ByteNGramEncoder:
    def __init__(self, n: int = 3, dim: int = 256, seed: int = 1234):
        self.n = n
        self.dim = dim
        rng = np.random.RandomState(seed)
        # Random projection matrix (+1/-1)
        self.R = rng.choice([-1.0, 1.0], size=(dim,), p=[0.5,0.5]).astype(np.float32)

    def _ngrams(self, text: str) -> List[bytes]:
        b = text.encode('utf-8', errors='ignore')
        return [b[i:i+self.n] for i in range(max(0,len(b)-self.n+1))]

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for ng in self._ngrams(text):
            h = int(hashlib.sha256(ng).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        # Signed random projection (SimHash-like)
        return vec * self.R
