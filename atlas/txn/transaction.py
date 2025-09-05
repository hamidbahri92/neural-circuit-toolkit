
import copy, json, numpy as np, time
from typing import Dict, Any, List, Tuple

class SimpleModel:
    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = {k: v.copy() for k,v in weights.items()}
    def state_dict(self):
        return {k: v.copy() for k,v in self.weights.items()}
    def load_state_dict(self, st: Dict[str, np.ndarray]):
        self.weights = {k: v.copy() for k,v in st.items()}

class CircuitTransaction:
    def __init__(self, model: SimpleModel):
        self.model = model
        self.checkpoints: List[Dict[str, np.ndarray]] = []
        self.history: List[Dict[str, Any]] = []
    def apply_row_delta(self, param: str, rows, delta):
        self.checkpoints.append(self.model.state_dict())
        W = self.model.weights[param]
        W2 = W.copy()
        W2[rows] += delta
        self.model.weights[param] = W2
        self.history.append({"param":param,"rows":list(rows),"delta_shape":list(delta.shape),"ts":time.time()})
    def rollback(self, steps: int = 1):
        if steps>len(self.checkpoints): steps=len(self.checkpoints)
        if steps==0: return
        st = self.checkpoints[-steps]
        self.model.load_state_dict(st)
        self.checkpoints = self.checkpoints[:-steps]
        self.history = self.history[:-steps]
    def commit(self):
        self.checkpoints = []
