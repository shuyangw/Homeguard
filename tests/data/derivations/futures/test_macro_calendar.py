"""Tests for macro event calendar loader."""
from datetime import date
from pathlib import Path

import pytest
import yaml

from src.data.derivations.futures.macro_calendar import (
    VALID_EVENT_TYPES,
    load_macro_calendar,
)


def _write_yaml(path: Path, event_type: str, dates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(
            {"event_type": event_type, "description": "test", "dates": dates},
            f,
        )


def test_loads_valid_fomc_calendar(tmp_path: Path):
    _write_yaml(tmp_path / "fomc.yaml", "fomc",
                ["2024-01-31", "2024-03-20", "2024-05-01"])
    dates = load_macro_calendar("fomc", calendar_dir=tmp_path)
    assert dates == [date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1)]


def test_invalid_event_type_raises():
    with pytest.raises(ValueError, match="unknown event_type"):
        load_macro_calendar("nonexistent")


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="calendar not found"):
        load_macro_calendar("fomc", calendar_dir=tmp_path)


def test_malformed_yaml_raises(tmp_path: Path):
    (tmp_path / "fomc.yaml").write_text("just_a_string", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_macro_calendar("fomc", calendar_dir=tmp_path)


def test_event_type_mismatch_raises(tmp_path: Path):
    _write_yaml(tmp_path / "fomc.yaml", "nfp", ["2024-01-05"])
    with pytest.raises(ValueError, match="event_type mismatch"):
        load_macro_calendar("fomc", calendar_dir=tmp_path)


def test_dates_returned_sorted(tmp_path: Path):
    _write_yaml(tmp_path / "fomc.yaml", "fomc",
                ["2024-03-20", "2024-01-31", "2024-05-01"])
    dates = load_macro_calendar("fomc", calendar_dir=tmp_path)
    assert dates == sorted(dates)


def test_all_event_types_valid():
    assert set(VALID_EVENT_TYPES) == {"fomc", "nfp", "cpi"}


# Integration tests against the committed real YAML files
def test_real_fomc_yaml_loads_covering_2010_2026():
    dates = load_macro_calendar("fomc")
    years = {d.year for d in dates}
    assert min(years) <= 2010
    assert max(years) >= 2026
    assert len(dates) > 100  # ~8 meetings/year x 17 years


def test_real_nfp_yaml_first_fridays():
    """NFP releases on first Friday of each month -- deterministic check."""
    dates = load_macro_calendar("nfp")
    # First Friday means day-of-week is Friday (4) AND day-of-month is 1-7
    for d in dates:
        assert d.weekday() == 4, f"NFP date {d} is not a Friday"
        assert d.day <= 7, f"NFP date {d} is not in first week"
    assert len(dates) == 17 * 12  # 17 years x 12 months


def test_real_cpi_yaml_loads():
    dates = load_macro_calendar("cpi")
    years = {d.year for d in dates}
    assert min(years) <= 2010
    assert max(years) >= 2026
    assert len(dates) == 17 * 12  # 17 years x 12 months
