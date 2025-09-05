
import numpy as np
from atlas.semantics.encoder import ByteNGramEncoder
from atlas.models.mock import MockBehaviorModel
from atlas.discover.min_pairs import minimal_pairs
from atlas.plan.glue_mock import plan_and_apply_behavior
from atlas.tests.invariants import regression_board, Invariant, evaluate_invariants

def test_end_to_end_mock_hedging():
    enc = ByteNGramEncoder(dim=128)
    base = MockBehaviorModel(enc, d_hidden=48)
    # Snapshot original
    import copy
    orig = copy.deepcopy(base)
    pos = ["maybe it could rain", "perhaps it will be cold", "it might snow later"]*8
    neg = ["clearly it will rain", "definitely cold front", "certainly snow is coming"]*8
    # Apply
    stat = plan_and_apply_behavior(base, "hedging", pos, neg, magnitude=0.5)
    assert stat["rows"] > 0
    # Verify hedging delta increased on minimal pairs (proxy)
    pairs = minimal_pairs("hedging", n=16)
    hed_before = []
    hed_after = []
    for a,b in pairs:
        hed_before.append(float(orig.behavior_scores([a])["hedging"].mean() - orig.behavior_scores([b])["hedging"].mean()))
        hed_after.append(float(base.behavior_scores([a])["hedging"].mean() - base.behavior_scores([b])["hedging"].mean()))
    assert np.mean(hed_after) >= np.mean(hed_before) - 1e-6
    # Invariants
    board = regression_board()
    policy = {
        "instruction_following": (board["instruction_following"], 0.9, True),
        "safety": (board["safety"], 0.9, True),
        "neutrality": (board["neutrality"], 0.8, False)
    }
    ok, results = evaluate_invariants(orig, base, policy)
    assert ok, f"Invariant breach: {results}"

if __name__ == "__main__":
    tests = [obj for name, obj in globals().items() if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"SUMMARY: {passed} passed, {failed} failed")
