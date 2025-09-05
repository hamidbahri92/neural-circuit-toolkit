
# RUN.md — Quickstart (in this environment)

```python
# 1) Run the end-to-end demo (discover → atlas → plan → apply → invariants)
from atlas.cli.demo import run_demo
report, orig, mod = run_demo()
print(report)

# 2) Inspect the Atlas manifest (circuits and DAG)
from atlas.cli.view import view_atlas
print(view_atlas("/mnt/data/universal_atlas/atlas/demo_atlas.json"))

# 3) Run the unit tests suite snippets (manually)
#    Each module has a __main__ block; you can also import and run test_ functions.
from atlas.tests import test_harness, test_discovery, test_end_to_end, test_atlas_planner_runtime, test_demo_cli
for mod in (test_harness, test_discovery, test_end_to_end, test_atlas_planner_runtime, test_demo_cli):
    for name in dir(mod):
        if name.startswith("test_"):
            getattr(mod, name)()
```

Artifacts written by the demo:
- docs/DEMO_REPORT.json — knobs, invariants, stability margins
- docs/DEMO_VIEW.txt — human-readable Atlas summary
