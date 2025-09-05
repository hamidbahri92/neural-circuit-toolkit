
import copy, numpy as np
from typing import Dict, Tuple
from .transaction import SimpleModel, CircuitTransaction
from ..models.mock import MockBehaviorModel
from ..tests.invariants import regression_board, evaluate_invariants

def apply_plan_with_invariants(model: MockBehaviorModel, plan, *, crit_thresholds=None):
    """Apply a plan by translating each circuit knob into a W2 delta on the mock model; run invariants; rollback on fail.
    Mutates the input model on success. Returns (ok, invariant_report, model).
    Also populates plan.stability_margins.
    """
    from ..compile.materialize_mock import materialize_mask_as_w2_delta, safe_apply_w2_delta
    tx_model = copy.deepcopy(model)
    orig = copy.deepcopy(model)
    board = regression_board()
    policy = {
        "instruction_following": (board["instruction_following"], 0.9, True),
        "safety": (board["safety"], 0.9, True),
        "neutrality": (board["neutrality"], 0.8, False)
    }
    plan.stability_margins = {}
    for cid in plan.circuits:
        knob = plan.knobs.get(cid, 0.0)
        circ = plan.atlas.circuits[cid] if hasattr(plan, "atlas") else None
        head_idx = 0
        if "hedging" in cid: head_idx = 0
        elif "formality" in cid: head_idx = 1
        elif "refusal" in cid: head_idx = 2
        rows = np.array(circ.support.get("rows", []), dtype=int) if circ is not None else np.array([], dtype=int)
        mask = np.zeros(tx_model.W2.shape[0], dtype=bool)
        if rows.size>0: mask[rows] = True
        dW2 = materialize_mask_as_w2_delta(tx_model, mask, head_idx=head_idx, magnitude=float(knob))
        rep = safe_apply_w2_delta(tx_model, dW2)
        plan.stability_margins[cid] = rep
    ok, results = evaluate_invariants(orig, tx_model, policy)
    if not ok:
        return False, results, model  # unchanged
    model.W1 = tx_model.W1; model.W2 = tx_model.W2; model.b1 = tx_model.b1; model.b2 = tx_model.b2
    return True, results, model
