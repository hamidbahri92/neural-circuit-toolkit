# Universal Atlas — Complete Documentation

Generated: 2025-09-01T14:13:27.342096Z

## Executive Summary

Universal Atlas is a deterministic toolkit for discovering, composing, and safely applying **sparse circuit edits** to neural models. 
It replaces per-model, per-run statistical probing with **gauge-invariant subspace alignment** and a **composition grammar** over circuits, 
backstopped by invariant tests and transactional rollback. This repo demonstrates the full pipeline on a mock model right here.

## 5W + How

**What**: A library to *discover → align → plan → apply → verify* behavioral circuits with stability and invariants.  
**Why**: To make behavior control portable, auditable, and cheap by operating in universal subspaces instead of ad hoc prompting.  
**Who**: Researchers and engineers who need programmable model behavior with safety rails.  
**Where**: Code under `atlas/`, blobs under `blobs/`, docs in `docs/`.  
**How**: Balanced contrast sets + minimal pairs → differential salience + sparsification → circuit diffs → hierarchical planning 
with interference prediction and knob solving → spectrally safe application → invariants + rollback → evolution.

## Live Demo Report
```json
{
  "ok": true,
  "invariants": {
    "instruction_following": 0.9999968209188915,
    "safety": 0.9999994225089154,
    "neutrality": 0.9999988774848961
  },
  "knobs": {
    "behavior/hedging@v1": 0.0,
    "behavior/formality@v1": 0.2
  },
  "leaves": [
    "behavior/hedging@v1",
    "behavior/formality@v1"
  ]
}
```

## Atlas Manifest View
```
Atlas version: v0.1 | family: mock_residual
Circuits:
  - behavior/formality@v1 [behavioral] rows=21
  - behavior/hedging@v1 [behavioral] rows=12
DAG:
  persona/weather_writer@v1 -> behavior/hedging@v1
  persona/weather_writer@v1 -> behavior/formality@v1
```

---
## System Design (from first principles)

### Data Contracts
- **CircuitDiff**: typed sparse edit with `support` (layer/type/rows/cols), `basis_blob` (content-addressed), and `deltas` (differential knobs). 
- **AtlasManifest**: versioned store of circuits, projection blobs, DAG for composition, and lineage.
- **Plan**: ordered circuit ids + knob magnitudes + predicted interference + stability margins.
- **Patch**: reversible tensor deltas (content-addressed) with audit metadata.

### Algorithms (sketch)
- **Ground Truth**: balance contrast sets on confounders (length, punctuation, semantic projection), verify with minimal pairs, then mine.
- **Discovery**: compute `sal = (E[H_pos] - E[H_neg]) ⊙ |∂y/∂H|`, prune to keep ≥80% effect, certify via activation patching, compute isolation.
- **Alignment**: whiten features per layer and compute Procrustes rotation; add simple dynamics fingerprints.
- **Planning**: DAG resolve, heuristic interference prediction (support overlap + effect cosine), knob solver to meet targets with penalties.
- **Stability**: cap spectral norm of edited matrices; compute utilization vs. budget and expose per-circuit margins.
- **Verification**: invariants (instruction-following, safety, neutrality) as hard gates; transactional rollback on breach.

### Implementation Highlights
- Content-addressed blobs (`sha256:`) for bases and patches; HMAC signatures for manifests.
- GPU-free mock model for deterministic tests; easy to swap with real LLM blocks.
- Transactional runtime with copy-on-write + audit trail; plan carries stability margins for ops visibility.

---
## How-To (in this notebook)

See `RUN.md` for exact commands. Typical flow:
1. `run_demo()` — creates a mini-Atlas with hedging↑, formality↑, refusal↑ and applies a safe plan.
2. Inspect `docs/DEMO_REPORT.json` and `docs/DEMO_VIEW.txt` for knobs, invariants, and spectral margins.
3. Read `docs/MASTER_SPEC.md` for design details.

---
## Test Matrix (live snapshot)

# Universal Atlas — Test Report

| Module | Test | Result |
|---|---|---|
| `atlas.tests.test_harness` | `test_apply_lowrank_and_norm` | PASS |
| `atlas.tests.test_harness` | `test_content_address_store_and_load` | PASS |
| `atlas.tests.test_harness` | `test_numpy_blob_roundtrip` | PASS |
| `atlas.tests.test_harness` | `test_procrustes_identity` | PASS |
| `atlas.tests.test_harness` | `test_transaction_apply_and_rollback` | PASS |
| `atlas.tests.test_harness` | `test_whiten_shapes` | PASS |
| `atlas.tests.test_discovery` | `test_confounders_and_balancing` | PASS |
| `atlas.tests.test_discovery` | `test_discovery_pipeline_mock` | PASS |
| `atlas.tests.test_discovery` | `test_minimal_pairs` | PASS |
| `atlas.tests.test_end_to_end` | `test_end_to_end_mock_hedging` | PASS |
| `atlas.tests.test_atlas_planner_runtime` | `test_plan_and_apply_rollback_on_failure` | PASS |
| `atlas.tests.test_atlas_planner_runtime` | `test_plan_and_apply_success` | PASS |
| `atlas.tests.test_demo_cli` | `test_run_demo_cli` | FAIL:  |

---
## Code Tree
```
./
atlas/
  __init__.py
  demo_atlas.json
  mini_atlas.json
  core/
    __init__.py
    atlas_store.py
    hierarchy.py
    spec.py
  utils/
    __init__.py
    utils.py
  semantics/
    __init__.py
    encoder.py
  discover/
    __init__.py
    contrast.py
    min_pairs.py
    mine.py
    to_circuit.py
  align/
    __init__.py
    alignment.py
  plan/
    __init__.py
    glue_mock.py
    knob_solver.py
    planner.py
    planner_obj.py
  compile/
    __init__.py
    apply.py
    materialize_mock.py
    stability.py
  txn/
    __init__.py
    runtime.py
    transaction.py
  tests/
    __init__.py
    invariants.py
    test_atlas_planner_runtime.py
    test_demo_cli.py
    test_discovery.py
    test_end_to_end.py
    test_harness.py
    test_prompts.py
  models/
    __init__.py
    mock.py
  cli/
    demo.py
    view.py
blobs/
docs/
  DEMO_REPORT.json
  MASTER_SPEC.md
  README_step1.md
  README_step2.md
  README_step3.md
  README_step4.md
  REPORT.md
```

---
## Extending to Real Models

- Expose per-block activations and weights in your model wrapper; mirror the interfaces used by the mock.
- Implement low-rank / row–col materialization for Attention Q/K/V/OV and MLP blocks, then route `CircuitDiff` supports appropriately.
- Replace heuristic interference with a learned predictor (GNN over support graphs) trained on your composition telemetry.
- Expand invariants to match your domain; keep critical gates strict and run canaries with immediate rollback.
