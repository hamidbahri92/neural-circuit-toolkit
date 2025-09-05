
import numpy as np
from atlas.semantics.encoder import ByteNGramEncoder
from atlas.discover.contrast import confounder_features, balance_weights, mmd2
from atlas.discover.min_pairs import minimal_pairs
from atlas.models.mock import MockBehaviorModel
from atlas.discover.mine import differential_salience, iterative_prune_preserve, activation_patching_verify, isolation_score

def test_confounders_and_balancing():
    enc = ByteNGramEncoder(dim=128)
    pos = ["maybe it could rain", "perhaps it will be cold", "it might snow later"]*5
    neg = ["clearly it will rain", "definitely cold front", "certainly snow is coming"]*5
    Xp = confounder_features(pos, enc); Xn = confounder_features(neg, enc)
    wpos, wneg = balance_weights(pos, neg, enc)
    m_before = mmd2((Xp, np.ones(len(pos))), (Xn, np.ones(len(neg))))
    m_after = mmd2((Xp, wpos), (Xn, wneg))
    assert m_after <= m_before + 1e-5

def test_minimal_pairs():
    pairs = minimal_pairs("hedging", n=8)
    assert len(pairs)==8 and all(isinstance(p, tuple) and len(p)==2 for p in pairs)

def test_discovery_pipeline_mock():
    enc = ByteNGramEncoder(dim=128)
    model = MockBehaviorModel(enc, d_hidden=32)
    pos = ["maybe it could rain", "perhaps it will be cold", "it might snow later"]*8
    neg = ["clearly it will rain", "definitely cold front", "certainly snow is coming"]*8
    sal = differential_salience(model, "hedging", pos, neg)["H1"]
    mask = iterative_prune_preserve(model, "hedging", pos, neg, sal, keep_frac=0.2)
    assert mask.sum() > 0
    verify = activation_patching_verify(model, "hedging", pos, neg, mask)
    assert verify["delta"] > 0.0
    iso = isolation_score(model, "hedging", mask, n_pairs=16)
    assert iso >= 0.5  # weak floor for mock model

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
