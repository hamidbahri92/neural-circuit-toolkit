
# Universal Atlas — Step 2: Ground Truth & Discovery (Mock Model)

This step introduces:
- Confounder mining and simple balancing weights for contrastive sets.
- Minimal-pair generators for three behaviors (hedging, formality, refusal).
- A tiny GPU-free mock model (`MockBehaviorModel`) that exposes activations and gradients.
- Automated circuit discovery: salience → iterative pruning → activation patching verify.
- An isolation score over minimal pairs.

Run tests to validate the pipeline: `python atlas/tests/test_discovery.py`.
