
import numpy as np
from typing import Dict, Tuple
from .planner import predict_interference
from ..discover.mine import differential_salience, iterative_prune_preserve
from ..compile.materialize_mock import materialize_mask_as_w2_delta, safe_apply_w2_delta
from ..models.mock import MockBehaviorModel

def plan_and_apply_behavior(model: MockBehaviorModel, behavior: str, pos, neg, magnitude: float = 0.1) -> Dict[str, float]:
    sal = differential_salience(model, behavior, pos, neg)["H1"]
    mask = iterative_prune_preserve(model, behavior, pos, neg, sal, keep_frac=0.2)
    head_idx = {"hedging":0,"formality":1,"refusal":2}[behavior]
    dW2 = materialize_mask_as_w2_delta(model, mask, head_idx, magnitude=magnitude)
    gamma = safe_apply_w2_delta(model, dW2)
    return {"rows": int(mask.sum()), "gamma": gamma}
