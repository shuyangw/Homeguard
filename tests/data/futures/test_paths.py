from pathlib import Path
import src.data.futures.paths as paths


def test_paths_point_at_consolidated_layout(monkeypatch):
    monkeypatch.setattr(paths, "get_local_storage_dir", lambda: Path("/data"))
    assert paths.continuous_1min_dir() == Path("/data/futures/databento/1min")
    assert paths.per_contract_1min_dir() == Path("/data/futures/databento/per_contract_1min")
    assert paths.statistics_dir() == Path("/data/futures/databento/statistics")
    assert paths.definitions_dir() == Path("/data/futures/definitions")
    assert paths.roll_calendar_dir() == Path("/data/futures/roll_calendar")
