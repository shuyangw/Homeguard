import datetime as dt

import pandas as pd
import pytest

from src.data.fx_rates import build_rate_diff_panel, load_fx_rate_panel
from src.data import fx_rates


def test_rate_diff_base_minus_quote():
    idx = pd.Index([dt.date(2024, 1, 2), dt.date(2024, 1, 3)])
    rates = pd.DataFrame(
        {"EUR": [0.04, 0.04], "USD": [0.053, 0.053], "JPY": [0.001, 0.001]}, index=idx)
    diff = build_rate_diff_panel(["EURUSD", "USDJPY"], rates)
    # EURUSD: r_EUR - r_USD = 0.04 - 0.053
    assert diff["EURUSD"].iloc[0] == pytest.approx(0.04 - 0.053)
    # USDJPY: r_USD - r_JPY = 0.053 - 0.001
    assert diff["USDJPY"].iloc[0] == pytest.approx(0.053 - 0.001)


def test_metals_base_rate_zero():
    idx = pd.Index([dt.date(2024, 1, 2)])
    rates = pd.DataFrame({"XAU": [0.0], "USD": [0.053]}, index=idx)
    diff = build_rate_diff_panel(["XAUUSD"], rates)
    # gold carry = 0 - r_USD (pure USD funding)
    assert diff["XAUUSD"].iloc[0] == pytest.approx(-0.053)


def _write_fred(root, series_id, dates, values):
    d = root / "alt_data" / "fred" / series_id
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "value": values}).to_parquet(d / "daily.parquet")


def test_load_fx_rate_panel_percent_to_decimal(monkeypatch, tmp_path):
    monkeypatch.setattr(fx_rates, "get_local_storage_dir", lambda: str(tmp_path))
    idx = pd.Index([dt.date(2024, 1, 2), dt.date(2024, 1, 3)])
    _write_fred(tmp_path, "DFF", [dt.date(2024, 1, 2), dt.date(2024, 1, 3)], [5.33, 5.33])
    panel = load_fx_rate_panel(["USD"], idx)
    assert panel["USD"].iloc[0] == pytest.approx(0.0533)  # percent -> decimal


def test_load_fx_rate_panel_metals_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(fx_rates, "get_local_storage_dir", lambda: str(tmp_path))
    idx = pd.Index([dt.date(2024, 1, 2)])
    panel = load_fx_rate_panel(["XAU"], idx)
    assert panel["XAU"].iloc[0] == 0.0


def test_load_fx_rate_panel_missing_file_is_zero_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(fx_rates, "get_local_storage_dir", lambda: str(tmp_path))
    idx = pd.Index([dt.date(2024, 1, 2)])
    panel = load_fx_rate_panel(["USD"], idx)  # DFF mapped but no file written
    assert panel["USD"].iloc[0] == 0.0
