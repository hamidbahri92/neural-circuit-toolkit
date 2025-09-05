
# Universal Atlas — Step 4: Hierarchical Atlas + Planner + Runtime

This step adds:
- `AtlasStore` to create/save/load a manifest with circuits and a DAG.
- Conversion from discovered masks to `CircuitDiff` with differential knobs.
- A `Planner` that builds a `Plan` and estimates interference risk.
- A runtime that applies a plan on the mock model with invariant gates and rollback.
- Tests that build a mini-Atlas, plan+apply, and verify success + rollback behavior.
