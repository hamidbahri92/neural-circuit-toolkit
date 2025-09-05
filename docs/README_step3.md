
# Universal Atlas — Step 3: Invariants, Stability, and E2E (Mock)

Adds:
- Stability guards (spectral cap + Lyapunov-style scaling hook).
- Invariant harness with a small regression board proxy (instruction-following, safety, neutrality).
- Planner glue to discover a mask and apply a safe W2 delta on the mock model.
- End-to-end test that checks behavior delta and that invariants hold after edit.

Run: `python atlas/tests/test_end_to_end.py`.
