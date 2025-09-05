
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from .planner import predict_interference
from ..core.spec import AtlasManifest, CircuitDiff

def head_index_from_id(cid: str) -> int:
    if "hedging" in cid: return 0
    if "formality" in cid: return 1
    if "refusal" in cid: return 2
    return 0

def proxy_effect_size(c: CircuitDiff) -> float:
    # Use number of rows as effect-size proxy
    rows = c.support.get("rows", [])
    return float(len(rows)) if rows else 1.0

def predict_plan_risk(atlas: AtlasManifest, circuit_ids: List[str]) -> float:
    risk = 0.0
    for i in range(len(circuit_ids)):
        for j in range(i+1, len(circuit_ids)):
            A = atlas.circuits[circuit_ids[i]]
            B = atlas.circuits[circuit_ids[j]]
            rep = predict_interference(A, B)
            risk = max(risk, rep.risk)
    return float(risk)

def solve_knobs(atlas: AtlasManifest, circuit_ids: List[str], targets: Dict[str, float], max_mag: float = 0.8) -> Dict[str, float]:
    """Grid-search small set of knob magnitudes to approximately meet targets per head while minimizing interference risk.
    targets: map of behavior keyword -> desired signed strength (e.g., {'hedging': +1.0, 'formality': +0.5})
    """
    # Candidate magnitudes
    mags = np.linspace(0.0, max_mag, num=5)
    best = None; best_obj = 1e9
    # Build mapping from circuit to target head
    head_map = {cid: head_index_from_id(cid) for cid in circuit_ids}
    for combo in np.ndindex(*(len(mags),)*len(circuit_ids)):
        knobs = {cid: float(mags[i]) for i, cid in enumerate(circuit_ids)}
        # Objective: squared error to targets using proxy effects + penalty for interference
        head_effect = {0:0.0,1:0.0,2:0.0}
        for cid, m in knobs.items():
            h = head_map[cid]
            eff = proxy_effect_size(atlas.circuits[cid]) * m
            head_effect[h] += eff
        err = 0.0
        for k, t in targets.items():
            h = 0 if k.startswith("hedg") else 1 if k.startswith("forma") else 2 if k.startswith("refus") else 0
            err += (head_effect[h] - t)**2
        # Interference penalty
        risk = predict_plan_risk(atlas, circuit_ids)
        obj = err + 0.2 * risk
        if obj < best_obj:
            best_obj = obj; best = knobs
    return best or {cid: 0.0 for cid in circuit_ids}
