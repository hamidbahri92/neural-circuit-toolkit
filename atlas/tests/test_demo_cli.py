
from atlas.cli.demo import run_demo

def test_run_demo_cli():
    report, orig, mod = run_demo()
    assert isinstance(report, dict) and 'ok' in report
    assert report['ok'] is True
    # Expect three leaves in persona
    assert len(report['leaves']) == 3
    # Stability margins present
    assert isinstance(report.get('stability_margins', {}), dict)
    # Nonzero knobs for at least two behaviors
    assert sum(1 for v in report['knobs'].values() if v > 0) >= 2
