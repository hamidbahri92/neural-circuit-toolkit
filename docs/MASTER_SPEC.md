
# Universal Atlas — Master Spec (Draft)

## What
A deterministic, GPU-native toolkit for discovering, aligning, composing, and safely applying **sparse circuit edits** to language models, with tests and reversibility.

## Why
Per-model discovery is wasteful and brittle. By operating in aligned, gauge-invariant subspaces and at the **circuit** level with safety rails, we get predictable, portable control at tiny cost.

## Who
Designed for researchers and engineers who need **programmable behavior** and auditable edits. Works here with a mock model; intended to plug into real LLMs.

## Where
Code lives under `atlas/` with clear modules: discover, align, plan, compile, txn, tests. Artifacts live under `blobs/` and `docs/`.

## How (first principles overview)
1. **Ground truth**: we balance contrast sets and gate discovery on counterfactual **minimal pairs**.
2. **Discovery**: compute differential salience, sparsify while preserving behavior, certify causality via activation patching; convert to `CircuitDiff`.
3. **Alignment**: (mock here) whitening + Procrustes & dynamics fingerprints to map atlas bases to model bases.
4. **Planning**: resolve dependencies, predict interference heuristically, solve knob magnitudes.
5. **Application**: materialize as low-rank/row edits with **spectral caps** and Lyapunov scaling; run invariants; **transactionally commit or rollback**.
6. **Verification**: unit, integration, regression boards with semantic and invariant gates.

## Math sketch (mock level)
- Differential salience: `sal = (E[H1_pos] - E[H1_neg]) ⊙ |∂y_b/∂H1|` for behavior `b`.
- Sparsification: greedy keep top-k units s.t. delta on behavior `Δ_b(mask)` remains ≥ 0.8 of baseline delta.
- Stability: cap `||W+Δ||₂ ≤ β||W||₂`, and scale to keep `ρ(A0+ΔA) ≤ margin` in the local subspace.

## Code map (selected)
- `atlas/discover/contrast.py`: confounders + balancing + MMD.
- `atlas/discover/mine.py`: salience → prune → patch verify + isolation score.
- `atlas/discover/to_circuit.py`: mask → `CircuitDiff` (with differential knob).
- `atlas/core/spec.py`: data contracts + content-addressed blob store.
- `atlas/core/atlas_store.py`: create/load/save an Atlas manifest.
- `atlas/plan/planner_obj.py`: build a `Plan` from a high-level behavior id.
- `atlas/compile/stability.py`: spectral cap + Lyapunov scaling.
- `atlas/txn/runtime.py`: apply a plan under invariant gates, commit/rollback.

## Next steps (real models)
- Replace mock materialization with attention/MLP row/col and low-rank patches.
- Add alignment caches and per-layer projections.
- Train an interference predictor and expand invariants.


---
### Hierarchical composition
We tag circuits by level (atomic→primitive→composite→behavioral→persona) and maintain a DAG. `decompose()` returns leaf circuits; plans are constructed over leaves for minimal edits.

### Knob solver
A pragmatic solver sweeps knob magnitudes to roughly match target behavior strengths using a proxy effect size (support cardinality) and a penalty for predicted interference. This is deliberately simple-but-deterministic for the mock; in real deployments, swap in a small QP or bilevel solver with empirical effect matrices.

### Demo usage
See `atlas/cli/demo.py::run_demo()` for end-to-end discovery→atlas→plan→apply with invariants. It builds a persona bundle that composes hedging↑ and formality↑ and applies it safely.


---
## How to run here
- Run `from atlas.cli.demo import run_demo; report, orig, mod = run_demo()` to execute discovery→atlas→plan→apply with invariants.
- Inspect the manifest with `from atlas.cli.view import view_atlas; print(view_atlas("/mnt/data/universal_atlas/atlas/demo_atlas.json"))`.
- See `docs/REPORT.md` for a live test matrix.

## Extending to real models (sketch)
- Swap `MockBehaviorModel` for a real module that exposes forward activations and block-level weights.
- Replace `materialize_mask_as_w2_delta` with low-rank/row-col patches on attention/MLP blocks.
- Integrate per-layer whitening/Procrustes and a learned fingerprint map; attach to `Planner` for alignment-aware plans.
- Keep invariants as hard gates; widen the regression board to your domains.
