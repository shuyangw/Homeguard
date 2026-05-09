"""Smoke test: GC density >700 bars/day (bug-fix smoking gun)."""
import pytest

from src.settings import get_local_storage_dir


def _gc_available() -> bool:
    root = get_local_storage_dir()
    return (root / "futures_1min" / "symbol=GC").exists()


@pytest.mark.skipif(not _gc_available(), reason="GC data not present locally")
def test_gc_density_above_threshold():
    import src.data.validation.futures.checks  # noqa: F401
    from src.data.validation.core.runner import ValidationRunner

    runner = ValidationRunner(
        domain="futures", layer=2, check_filter="density_GC", mode="quarterly",
    )
    report = runner.run()
    matching = [r for r in report.results if "density_GC" in r.name]
    assert matching, "no density_GC check ran"
    r = matching[0]
    assert r.details.get("rows_per_day", 0) > 700, (
        f"GC density {r.details.get('rows_per_day')} bars/day fell below 700 "
        f"-- the v.0/c.0 bug may have regressed."
    )
