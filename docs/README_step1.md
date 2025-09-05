
# Universal Atlas — Step 1: Core Skeleton

This step lays down the core data contracts, content-addressed blob store,
alignment utilities (whitening + Procrustes), simple compile/apply routines,
a transactional apply/rollback layer, and a basic semantic encoder.

## What exists
- `atlas/core/spec.py`: CircuitDiff, AtlasManifest, Plan, Patch; content-addressed blobs; HMAC signatures.
- `atlas/utils/utils.py`: SHA256 blobs, save/load .npy, whitening, orthogonal Procrustes.
- `atlas/semantics/encoder.py`: Byte-level n-gram encoder (`E_sem`) for tokenizer-agnostic probes.
- `atlas/align/alignment.py`: Whitening + Procrustes pipeline and a dynamics fingerprint shim.
- `atlas/compile/apply.py`: Numpy low-rank and row-edit application + spectral norm estimate.
- `atlas/txn/transaction.py`: Simple model container + transactional apply/rollback.
- `atlas/tests/test_harness.py`: Unit tests you can run directly in this environment.

## Next
- Discovery with balanced contrast sets and minimal pairs.
- Invariant test harness and regression board.
- Planner with DAG + interference predictor (initial heuristic).
