
from __future__ import annotations
import copy, json
from typing import Dict
from atlas.semantics.encoder import ByteNGramEncoder
from atlas.models.mock import MockBehaviorModel
from atlas.discover.mine import differential_salience, iterative_prune_preserve
from atlas.discover.to_circuit import mask_to_circuit
from atlas.core.atlas_store import AtlasStore
from atlas.core.hierarchy import add_level_tag, decompose
from atlas.plan.planner_obj import Planner
from atlas.plan.knob_solver import solve_knobs
from atlas.txn.runtime import apply_plan_with_invariants
from atlas.tests.invariants import regression_board

def run_demo():
    enc = ByteNGramEncoder(dim=128)
    model = MockBehaviorModel(enc, d_hidden=64)
    orig = copy.deepcopy(model)
    # Discover hedging
    pos_h = ["maybe it could rain", "perhaps it will be cold", "it might snow later"]*8
    neg_h = ["clearly it will rain", "definitely cold front", "certainly snow is coming"]*8
    sal_h = differential_salience(model, "hedging", pos_h, neg_h)["H1"]
    mask_h = iterative_prune_preserve(model, "hedging", pos_h, neg_h, sal_h, keep_frac=0.2)
    # Discover formality
    pos_f = ["therefore precipitation may increase", "moreover winds will shift", "thus expect rainfall"]*8
    neg_f = ["yeah it might rain", "btw wind changes", "kinda rainy later"]*8
    sal_f = differential_salience(model, "formality", pos_f, neg_f)["H1"]
    mask_f = iterative_prune_preserve(model, "formality", pos_f, neg_f, sal_f, keep_frac=0.2)
    # Discover refusal
    pos_r = ["I cannot comply with that request.", "I won't do that.", "That would be inappropriate."]*8
    neg_r = ["Sure, here's how.", "Absolutely, let's do it.", "Yes, proceeding."]*8
    sal_r = differential_salience(model, "refusal", pos_r, neg_r)["H1"]
    mask_r = iterative_prune_preserve(model, "refusal", pos_r, neg_r, sal_r, keep_frac=0.2)
    # Build atlas
    store = AtlasStore("/mnt/data/universal_atlas/atlas/demo_atlas.json").new(family="mock_residual")
    c_h = add_level_tag(mask_to_circuit("behavior/hedging@v1", layer=1, rows_mask=mask_h, behavior_axis="hedge"), "behavioral")
    c_f = add_level_tag(mask_to_circuit("behavior/formality@v1", layer=1, rows_mask=mask_f, behavior_axis="formal"), "behavioral")
    c_r = add_level_tag(mask_to_circuit("behavior/refusal@v1", layer=1, rows_mask=mask_r, behavior_axis="refuse"), "behavioral")
    store.add_circuit(c_h); store.add_circuit(c_f); store.add_circuit(c_r)
    store.add_edge("persona/weather_writer@v1", "behavior/hedging@v1")
    store.add_edge("persona/weather_writer@v1", "behavior/formality@v1")
    store.add_edge("persona/weather_writer@v1", "behavior/refusal@v1")
    store.save()
    atlas = store.manifest
    # Plan: target strengths
    planner = Planner(atlas)
    leaves = decompose(atlas, "persona/weather_writer@v1")
    knobs = solve_knobs(atlas, leaves, targets={"hedging": 0.6, "formality": 0.5, "refusal": 0.2}, max_mag=0.8)
    plan = planner.build_plan("persona/weather_writer@v1", magnitude=0.0)
    plan.circuits = leaves
    plan.knobs = knobs
    plan.atlas = atlas
    ok, inv, mod = apply_plan_with_invariants(model, plan)
    report = {
        "ok": ok,
        "invariants": inv,
        "knobs": knobs,
        "leaves": leaves,
        "stability_margins": plan.stability_margins
    }
    return report, orig, mod
