"""Ground-truth regime labelers (G1, G2, G3, G4) for the diagnostic.

Each labeler operates on a SPY+VIX panel indexed by date. None look ahead
unless explicitly noted (G2 is forward-looking and IN-SAMPLE ONLY -- not
used for any live decision).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import logger


def label_g1_drawdown_bear(
    panel: pd.DataFrame,
    threshold_pct: float = 10.0,
    lookback_days: int = 252,
) -> pd.Series:
    """G1: SPY drawdown from trailing N-day high exceeds threshold."""
    spy = panel['spy_close']
    rolling_peak = spy.rolling(lookback_days, min_periods=1).max()
    dd = (spy / rolling_peak - 1.0) * 100.0
    return (dd <= -threshold_pct).rename('g1_bear')


def label_g2_forward_window_bear(
    panel: pd.DataFrame,
    fwd_days: int = 30,
    ret_threshold: float = -0.05,
    vol_threshold: float = 0.25,
) -> pd.Series:
    """G2: forward 30-day SPY return < -5% AND forward vol > 25%.

    FORWARD-LOOKING. In-sample only. Not for live decisions.
    """
    spy = panel['spy_close']
    fwd_ret = spy.shift(-fwd_days) / spy - 1.0
    returns = spy.pct_change()
    # Forward 30d realized vol: compute rolling std over fwd_days, then
    # shift back so each row reflects the [t+1, t+fwd_days] window.
    fwd_vol = returns.rolling(fwd_days).std().shift(-fwd_days) * np.sqrt(252)
    labels = (fwd_ret < ret_threshold) & (fwd_vol > vol_threshold)
    return labels.fillna(False).rename('g2_bear')


def label_g3_vol_spike(
    panel: pd.DataFrame,
    vix_abs_threshold: float = 30.0,
    vix_5d_pct_threshold: float = 0.5,
) -> pd.Series:
    """G3: VIX > absolute threshold OR VIX rose > pct over trailing 5 days."""
    vix = panel['vix_close']
    above_abs = vix > vix_abs_threshold
    rolling_pct = (vix / vix.shift(5)) - 1.0
    rapid_rise = rolling_pct > vix_5d_pct_threshold
    return (above_abs | rapid_rise).fillna(False).rename('g3_vol_spike')


def label_g4_hand_curated(
    panel: pd.DataFrame,
    csv_path: Path,
) -> pd.DataFrame:
    """G4: read hand-curated event windows from CSV.

    Returns a DataFrame with columns ['g4_event', 'g4_event_type'].
    Days outside any event window have NaN values.
    """
    events = pd.read_csv(csv_path, parse_dates=['start_date', 'end_date'])
    g4 = pd.DataFrame(index=panel.index, columns=['g4_event', 'g4_event_type'])
    for _, row in events.iterrows():
        mask = (panel.index >= row['start_date']) & (panel.index <= row['end_date'])
        g4.loc[mask, 'g4_event'] = row['event_name']
        g4.loc[mask, 'g4_event_type'] = row['event_type']
    return g4


def build_ground_truth(
    panel: pd.DataFrame,
    csv_path: Path,
    output: Path,
) -> pd.DataFrame:
    """Compute all 4 labelers and write a combined Parquet."""
    g1 = label_g1_drawdown_bear(panel)
    g2 = label_g2_forward_window_bear(panel)
    g3 = label_g3_vol_spike(panel)
    g4 = label_g4_hand_curated(panel, csv_path)
    df = pd.DataFrame({
        'date': panel.index,
        'g1_bear': g1.values,
        'g2_bear': g2.values,
        'g3_vol_spike': g3.values,
        'g4_event': g4['g4_event'].values,
        'g4_event_type': g4['g4_event_type'].values,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    logger.info(f'[+] Wrote {output} ({len(df)} rows)')
    logger.info(f'    g1_bear: {df["g1_bear"].sum()} days')
    logger.info(f'    g2_bear: {df["g2_bear"].sum()} days')
    logger.info(f'    g3_vol_spike: {df["g3_vol_spike"].sum()} days')
    logger.info(f'    g4_event: {df["g4_event"].notna().sum()} days')
    return df


def main() -> int:
    panel_path = Path('diagnostics/data/spy_vix_2016_2026.parquet')
    csv_path = Path('config/diagnostics/regime_events_2017_2026.csv')
    output = Path('diagnostics/regime/ground_truth.parquet')

    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    panel = pd.read_parquet(panel_path)
    panel = panel.loc['2017-01-01':]

    build_ground_truth(panel, csv_path, output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
