
import numpy as np
from atlas.core.spec import CircuitDiff, AtlasManifest, content_address_store, content_address_load
from atlas.utils.utils import save_npy_blob, load_npy_blob, orthogonal_procrustes, whiten
from atlas.compile.apply import apply_lowrank, apply_row_edit, spectral_norm
from atlas.txn.transaction import SimpleModel, CircuitTransaction

def test_content_address_store_and_load():
    data = b'hello world'
    ref = content_address_store(data, suffix=".dat")
    back = content_address_load(ref)
    assert back == data

def test_numpy_blob_roundtrip():
    arr = np.random.randn(4,4).astype(np.float32)
    ref = save_npy_blob(arr, content_address_store)
    arr2 = load_npy_blob(ref, content_address_load)
    assert np.allclose(arr, arr2)

def test_procrustes_identity():
    A = np.eye(3)
    R = orthogonal_procrustes(A, A)
    assert np.allclose(R, np.eye(3), atol=1e-6)

def test_whiten_shapes():
    X = np.random.randn(100, 8).astype(np.float32)
    Xw, W = whiten(X)
    assert Xw.shape == X.shape and W.shape == (8,8)

def test_apply_lowrank_and_norm():
    W = np.random.randn(16,16).astype(np.float32)
    U = np.random.randn(16,4).astype(np.float32)
    V = np.random.randn(16,4).astype(np.float32)
    W2 = apply_lowrank(W, U, V)
    # Norm should not be NaN and change
    assert not np.isnan(W2).any()
    assert np.linalg.norm(W2 - W) > 0

def test_transaction_apply_and_rollback():
    W = np.zeros((10,10), dtype=np.float32)
    model = SimpleModel({"layer0": W})
    tx = CircuitTransaction(model)
    tx.apply_row_delta("layer0", rows=[1,3,5], delta=np.ones((3,10), dtype=np.float32))
    assert model.weights["layer0"][1].sum() == 10.0
    tx.rollback()
    assert np.allclose(model.weights["layer0"], W)
    tx.commit()

if __name__ == "__main__":
    # Run tests and print a simple report
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
