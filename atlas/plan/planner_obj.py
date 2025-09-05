
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .planner import dependency_plan, predict_interference
from ..core.spec import Plan, AtlasManifest

@dataclass
class Objective:
    target: float
    weight: float

class Planner:
    def __init__(self, atlas: AtlasManifest):
        self.atlas = atlas

    def build_plan(self, target_behavior_id: str, magnitude: float = 0.5) -> Plan:
        # Simple: decompose via DAG and set knob=magnitude for the leaf circuit
        order = dependency_plan(self.atlas.dag, target_behavior_id) if self.atlas.dag else [target_behavior_id]
        circuits = order
        knobs = {cid: magnitude for cid in circuits}
        # Heuristic interference risk
        risk = 0.0
        for i in range(len(circuits)):
            for j in range(i+1, len(circuits)):
                A = self.atlas.circuits[circuits[i]]
                B = self.atlas.circuits[circuits[j]]
                rep = predict_interference(A, B)
                risk = max(risk, rep.risk)
        return Plan(circuits=circuits, knobs=knobs, predicted_deltas=[], predicted_interference=risk, stability_margins={})
