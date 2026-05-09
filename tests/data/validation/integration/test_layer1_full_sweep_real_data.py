"""Layer 1 sweep against the real H:/Stock_Data/futures_*/ store.

Skipped automatically if the local storage isn't accessible.
"""
import pytest

from src.settings import get_local_storage_dir


def _local_storage_available() -> bool:
    root = get_local_storage_dir()
    return (root / "futures_1min" / "symbol=ES").exists()


@pytest.mark.skipif(not _local_storage_available(), reason="local futures store not present")
def test_layer1_sweep_passes():
    import src.data.validation.futures.checks  # noqa: F401  registers
    from src.data.validation.core.runner import ValidationRunner

    runner = ValidationRunner(domain="futures", layer=1, mode="quarterly")
    report = runner.run()
    assert report.critical_failures == 0, (
        f"Layer 1 has CRITICAL failures: {[r.name for r in report.results if not r.passed and r.severity.value == 'critical']}"
    )
