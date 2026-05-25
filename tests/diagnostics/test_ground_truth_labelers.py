"""TDD tests for scripts/diagnostics/ground_truth_labelers.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.ground_truth_labelers import (
    label_g1_drawdown_bear,
    label_g2_forward_window_bear,
    label_g3_vol_spike,
    label_g4_hand_curated,
    build_ground_truth,
)


def _synthetic_panel_with_drawdown() -> pd.DataFrame:
    """Build a panel with a known 25% drawdown event mimicking COVID.

    The drop window includes high-amplitude daily noise so realized vol
    exceeds 25% annualized, matching real-world crashes.
    """
    dates = pd.bdate_range('2020-01-01', '2020-12-31')
    n = len(dates)
    rng = np.random.default_rng(42)
    prices = np.full(n, 100.0)
    peak_idx = 30
    trough_idx = 60
    drop_n = trough_idx - peak_idx
    # Noisy drop: linear baseline plus daily shocks (~3% std => ~50% annualized).
    baseline = np.linspace(100, 75, drop_n)
    shocks = rng.normal(0.0, 0.03, drop_n)
    prices[peak_idx:trough_idx] = baseline * np.exp(np.cumsum(shocks) - np.cumsum(shocks)[-1] * np.arange(drop_n) / drop_n)
    prices[trough_idx:] = np.linspace(prices[trough_idx - 1], 90, n - trough_idx)
    vix = np.full(n, 15.0)
    vix[peak_idx:trough_idx] = np.linspace(15, 60, drop_n)
    return pd.DataFrame({
        'spy_close': prices,
        'vix_close': vix,
    }, index=dates)


def test_g1_drawdown_bear_fires_on_known_drawdown():
    panel = _synthetic_panel_with_drawdown()
    labels = label_g1_drawdown_bear(panel, threshold_pct=10.0, lookback_days=252)
    # Drawdown reaches -25% around day 60; G1_BEAR should fire there.
    assert labels.dtype == bool
    assert labels.loc[panel.index[60]] == True
    # Day 0: no drawdown yet (single price).
    assert labels.iloc[0] == False


def test_g2_forward_window_bear_uses_future_returns():
    panel = _synthetic_panel_with_drawdown()
    labels = label_g2_forward_window_bear(panel, fwd_days=30,
                                          ret_threshold=-0.05,
                                          vol_threshold=0.25)
    assert labels.dtype == bool
    # Day around peak should label True (forward 30d sees big drop + high vol).
    peak_t = panel.index[28]
    assert labels.loc[peak_t] == True
    # Very last days have no 30-day forward window -> False or NaN.
    assert labels.iloc[-1] == False


def test_g3_vol_spike_fires_on_vix_threshold():
    panel = _synthetic_panel_with_drawdown()
    labels = label_g3_vol_spike(panel, vix_abs_threshold=30.0, vix_5d_pct_threshold=0.5)
    # During the constructed VIX spike, label should fire.
    assert labels.dtype == bool
    # Mid-drawdown, VIX exceeded 30.
    mid_t = panel.index[55]
    assert labels.loc[mid_t] == True


def test_g4_hand_curated_assigns_event_types(tmp_path: Path):
    """G4 labels are populated from a CSV; verify event windows correctly mapped."""
    csv_path = tmp_path / 'events.csv'
    csv_path.write_text(
        'event_name,start_date,end_date,event_type\n'
        'test_dd,2020-02-01,2020-02-29,drawdown\n'
    )
    panel = _synthetic_panel_with_drawdown()
    labels = label_g4_hand_curated(panel, csv_path)
    assert isinstance(labels, pd.DataFrame)
    assert {'g4_event', 'g4_event_type'}.issubset(labels.columns)
    in_event = (panel.index >= '2020-02-01') & (panel.index <= '2020-02-29')
    assert (labels.loc[in_event, 'g4_event_type'] == 'drawdown').all()
    assert (labels.loc[~in_event, 'g4_event_type'].isna()).all()


def test_build_ground_truth_combines_all_four(tmp_path: Path):
    """End-to-end labeler emits one Parquet with all 4 labelers' columns."""
    csv_path = tmp_path / 'events.csv'
    csv_path.write_text(
        'event_name,start_date,end_date,event_type\n'
        'test_dd,2020-02-01,2020-02-29,drawdown\n'
    )
    panel = _synthetic_panel_with_drawdown()
    out = tmp_path / 'ground_truth.parquet'
    df = build_ground_truth(panel, csv_path, out)
    expected_cols = {
        'date', 'g1_bear', 'g2_bear', 'g3_vol_spike',
        'g4_event', 'g4_event_type',
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == len(panel)
