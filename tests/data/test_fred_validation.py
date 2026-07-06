from datetime import date

import pandas as pd
import pytest
from src.data.acquisition.plugins import fred_rates
from src.data.acquisition.plugins.fred_rates import (
    FREDRatesPlugin, validate_fred_series, FredValidationError,
)


def test_rejects_empty():
    with pytest.raises(FredValidationError):
        validate_fred_series(pd.Series(dtype=float), "TEST")


def test_rejects_implausible_rate():
    s = pd.Series([250.0, 300.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    with pytest.raises(FredValidationError):
        validate_fred_series(s, "TEST")


def test_accepts_plausible():
    s = pd.Series([3.5, 3.75], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    validate_fred_series(s, "TEST")  # no raise


def test_fetch_series_returns_error_dict_on_validation_gate_trip(tmp_path, monkeypatch):
    def fake_data_reader(series_id, source, start, end):
        return pd.DataFrame(
            {series_id: [250.0, 300.0]},
            index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
        )

    monkeypatch.setattr(fred_rates, "DataReader", fake_data_reader)

    plugin = FREDRatesPlugin(storage_root=tmp_path)
    result = plugin.fetch_series(
        "TEST", date(2020, 1, 1), date(2020, 2, 1), skip_existing=False
    )

    assert result["rows"] == 0
    assert result["error"] is not None
    assert result["out_path"] is None
    assert not (tmp_path / "fred" / "TEST" / "daily.parquet").exists()
