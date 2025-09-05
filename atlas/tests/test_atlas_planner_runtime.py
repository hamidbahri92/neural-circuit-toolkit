
import copy
from atlas.semantics.encoder import ByteNGramEncoder
from atlas.models.mock import MockBehaviorModel
from atlas.discover.min_pairs import minimal_pairs
from atlas.discover.mine import differential_salience, iterative_prune_preserve
from atlas.discover.to_circuit import mask_to_circuit
from atlas.core.atlas_store import AtlasStore
from atlas.plan.planner_obj import Planner
from atlas.txn.runtime import apply_plan_with_invariants

def build_mini_atlas(enc_dim=128, d_hidden=48):
    enc = ByteNGramEncoder(dim=enc_dim)
    model = MockBehaviorModel(enc, d_hidden=d_hidden)
    pos = ["maybe it could rain", "perhaps it will be cold", "it might snow later"]*8
    neg = ["clearly it will rain", "definitely cold front", "certainly snow is coming"]*8
    sal = differential_salience(model, "hedging", pos, neg)["H1"]
    mask = iterative_prune_preserve(model, "hedging", pos, neg, sal, keep_frac=0.2)
    circ = mask_to_circuit("behavior/hedging@v1", layer=1, rows_mask=mask, behavior_axis="hedge")
    store = AtlasStore("/mnt/data/universal_atlas/atlas/mini_atlas.json").new(family="mock_residual")
    store.add_circuit(circ); store.save()
    return store.manifest, model, pos, neg

def test_plan_and_apply_success():
    atlas, model, pos, neg = build_mini_atlas()
    planner = Planner(atlas)
    plan = planner.build_plan("behavior/hedging@v1", magnitude=0.4)
    plan.atlas = atlas
    # Snapshot before
    pairs = minimal_pairs("hedging", n=16)
    before = []
    for a,b in pairs:
        before.append(float(model.behavior_scores([a])["hedging"].mean() - model.behavior_scores([b])["hedging"].mean()))
    ok, inv, modified = apply_plan_with_invariants(model, plan)
    assert ok, f"Invariants failed: {inv}"
    after = []
    for a,b in pairs:
        after.append(float(modified.behavior_scores([a])["hedging"].mean() - modified.behavior_scores([b])["hedging"].mean()))
    avg_before = sum(before)/max(1,len(before))
    avg_after = sum(after)/max(1,len(after))
    assert avg_after >= avg_before - 1e-6

def test_plan_and_apply_rollback_on_failure():
    atlas, model, pos, neg = build_mini_atlas()
    planner = Planner(atlas)
    plan = planner.build_plan("behavior/hedging@v1", magnitude=5.0)
    plan.atlas = atlas
    ok, inv, maybe = apply_plan_with_invariants(model, plan)
    if not ok:
        from atlas.tests.invariants import regression_board
        board = regression_board()
        base_safety = board["safety"](model)
        new_safety = board["safety"](maybe)
        assert abs(base_safety - new_safety) < 1e-6
