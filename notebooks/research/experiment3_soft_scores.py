"""Experiment 3 -- Soft-score extraction from MarketRegimeDetector.

Three diagnostics that determine which WS-3 detector-improvement track is
correct:

(a) Event-study: BEAR_score trajectory in [-30, +10] trading days around
    each BEAR onset (argmax flip into BEAR). If the BEAR_score is elevated
    well before the argmax flips, the detector's signal is already there
    and argmax is suppressing it -- track WS-3c.

(b) Cross-correlation: Pearson r(BEAR_score_today, forward SPY drawdown
    over next h days) for h in {1, 5, 10, 20}. Tests whether the
    BEAR_score has predictive content independent of the argmax label.

(c) Threshold sweep: for tau in {0.2, 0.3, 0.4, 0.5}, median lag from
    BEAR_score >= tau (in the 30 days preceding onset) to (i) the argmax
    flip and (ii) the SPY drawdown trough in [-10, +10] window. Defines
    WS-3c's potential headroom: how many days earlier than the argmax
    would a threshold-based BEAR detector fire?

Decision criterion (final verdict):
  - median argmax_lag at tau=0.3 > 3.0 days       -> WS-3c
  - elif mean Pearson |r| at h=5d > 0.15 AND
        median argmax_lag at tau=0.3 <= 3.0 days  -> WS-3a
  - else                                          -> WS-3b

Inputs:
  diagnostics/regime/v0_scores/labels.parquet  (E3 soft-score replay)
  diagnostics/data/spy_vix_2016_2026.parquet   (SPY+VIX panel)

Outputs:
  diagnostics/regime/v0_scores/event_study_bear_score.csv
  diagnostics/regime/v0_scores/event_study_summary.csv
  diagnostics/regime/v0_scores/cross_correlation.csv
  diagnostics/regime/v0_scores/threshold_sweep.csv
  diagnostics/regime/v0_scores/verdict.txt

Run:
    PYTHONPATH=. python notebooks/research/experiment3_soft_scores.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import logger


SCORES_PATH = Path('diagnostics/regime/v0_scores/labels.parquet')
PANEL_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
OUT_DIR = Path('diagnostics/regime/v0_scores')

EVENT_WINDOW_PRE = 30
EVENT_WINDOW_POST = 10
TROUGH_WINDOW = 10
HORIZONS = (1, 5, 10, 20)
TAUS = (0.2, 0.3, 0.4, 0.5)


def load_scores() -> pd.DataFrame:
    df = pd.read_parquet(SCORES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    logger.info(f'[+] Loaded {len(df)} rows from {SCORES_PATH}')
    return df


def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    logger.info(f'[+] Loaded {len(panel)} rows from {PANEL_PATH}')
    return panel


def identify_bear_onsets(scores: pd.DataFrame) -> List[int]:
    """Index positions (in scores DataFrame) where regime flips into BEAR.

    A "BEAR onset" is a day where the previous day's argmax was NOT BEAR
    but today's IS BEAR.
    """
    regime = scores['regime'].values
    onsets = []
    for i in range(1, len(regime)):
        if regime[i] == 'BEAR' and regime[i - 1] != 'BEAR':
            onsets.append(i)
    logger.info(f'[+] Identified {len(onsets)} BEAR onsets')
    return onsets


def find_trough_index(spy_close: np.ndarray, onset_idx: int,
                      window: int = TROUGH_WINDOW) -> int:
    """Index of SPY local minimum in [onset-window, onset+window].

    Returned index is into the scores DataFrame (same alignment as
    onset_idx). The min is taken inclusive on both ends; ties resolve to
    the earliest occurrence.
    """
    lo = max(0, onset_idx - window)
    hi = min(len(spy_close) - 1, onset_idx + window)
    segment = spy_close[lo:hi + 1]
    rel_min = int(np.argmin(segment))
    return lo + rel_min


def compute_event_study(scores: pd.DataFrame, onsets: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-event BEAR_score trajectory + aggregated median/IQR summary."""
    bear_score = scores['score_BEAR'].values
    spy_close = scores['spy_close'].values
    dd = scores['spy_drawdown_from_252d_high'].values
    dates = scores['date'].values
    n = len(scores)

    per_event_rows = []
    for event_id, onset_idx in enumerate(onsets):
        for rel in range(-EVENT_WINDOW_PRE, EVENT_WINDOW_POST + 1):
            j = onset_idx + rel
            if j < 0 or j >= n:
                continue
            per_event_rows.append({
                'event_id': event_id,
                'onset_date': pd.Timestamp(dates[onset_idx]).date(),
                'relative_day': rel,
                'bear_score': float(bear_score[j]),
                'spy_close': float(spy_close[j]),
                'drawdown_from_252d_high': float(dd[j]),
            })

    per_event_df = pd.DataFrame(per_event_rows)

    summary_rows = []
    for rel in range(-EVENT_WINDOW_PRE, EVENT_WINDOW_POST + 1):
        bucket = per_event_df[per_event_df['relative_day'] == rel]['bear_score']
        bucket_clean = bucket.dropna()
        if len(bucket_clean) == 0:
            continue
        summary_rows.append({
            'relative_day': rel,
            'median_bear_score': float(bucket_clean.median()),
            'p25_bear_score': float(bucket_clean.quantile(0.25)),
            'p75_bear_score': float(bucket_clean.quantile(0.75)),
            'n_events': int(len(bucket_clean)),
        })
    summary_df = pd.DataFrame(summary_rows)

    return per_event_df, summary_df


def compute_cross_correlation(scores: pd.DataFrame) -> pd.DataFrame:
    """Pearson r(BEAR_score_today, forward SPY drawdown over next h days)."""
    spy_close = scores['spy_close'].values
    bear_score = scores['score_BEAR'].values
    n = len(scores)

    rows = []
    for h in HORIZONS:
        fwd_dd = np.full(n, np.nan)
        for i in range(n - h):
            window = spy_close[i + 1:i + 1 + h]
            if len(window) == 0:
                continue
            fwd_dd[i] = float(window.min() / spy_close[i] - 1.0)
        mask = ~np.isnan(bear_score) & ~np.isnan(fwd_dd)
        x = bear_score[mask]
        y = fwd_dd[mask]
        if len(x) < 2:
            r, p = float('nan'), float('nan')
        else:
            r_val, p_val = stats.pearsonr(x, y)
            r, p = float(r_val), float(p_val)
        rows.append({
            'horizon_days': h,
            'pearson_r': r,
            'n_obs': int(mask.sum()),
            'p_value': p,
        })
    return pd.DataFrame(rows)


def find_first_tau_crossing(bear_score: np.ndarray, onset_idx: int,
                            tau: float, lookback: int = 30) -> Optional[int]:
    """First index in [onset-lookback, onset-1] where BEAR_score >= tau."""
    lo = max(0, onset_idx - lookback)
    for j in range(lo, onset_idx):
        if not np.isnan(bear_score[j]) and bear_score[j] >= tau:
            return j
    return None


def compute_threshold_sweep(scores: pd.DataFrame, onsets: List[int]) -> pd.DataFrame:
    bear_score = scores['score_BEAR'].values
    spy_close = scores['spy_close'].values
    dates = scores['date'].values

    rows = []
    for tau in TAUS:
        argmax_lags = []
        trough_lags = []
        for onset_idx in onsets:
            cross_idx = find_first_tau_crossing(bear_score, onset_idx, tau)
            if cross_idx is None:
                continue
            trough_idx = find_trough_index(spy_close, onset_idx)
            argmax_lag = onset_idx - cross_idx
            trough_lag = trough_idx - cross_idx
            argmax_lags.append(argmax_lag)
            trough_lags.append(trough_lag)
        if not argmax_lags:
            rows.append({
                'tau': tau,
                'median_argmax_lag': float('nan'),
                'mean_argmax_lag': float('nan'),
                'median_trough_lag': float('nan'),
                'mean_trough_lag': float('nan'),
                'n_events_crossed': 0,
            })
        else:
            rows.append({
                'tau': tau,
                'median_argmax_lag': float(np.median(argmax_lags)),
                'mean_argmax_lag': float(np.mean(argmax_lags)),
                'median_trough_lag': float(np.median(trough_lags)),
                'mean_trough_lag': float(np.mean(trough_lags)),
                'n_events_crossed': len(argmax_lags),
            })
        _ = dates  # silence: dates not used here but kept for future debug
    return pd.DataFrame(rows)


def compute_onset_to_trough_gap(scores: pd.DataFrame, onsets: List[int]) -> float:
    """Mean of (onset_idx - trough_idx) across events.

    Negative -> detector fires AFTER the trough on average (consistent
    with V12's mean -3.42).
    """
    spy_close = scores['spy_close'].values
    gaps = []
    for onset_idx in onsets:
        trough_idx = find_trough_index(spy_close, onset_idx)
        gaps.append(onset_idx - trough_idx)
    return float(np.mean(gaps)) if gaps else float('nan')


def render_verdict(threshold_df: pd.DataFrame, xcorr_df: pd.DataFrame,
                   gap_days: float, n_onsets: int) -> str:
    """Apply the decision criterion and produce the verdict text."""
    tau_row = threshold_df[threshold_df['tau'] == 0.3]
    if tau_row.empty:
        median_argmax_lag = float('nan')
    else:
        median_argmax_lag = float(tau_row['median_argmax_lag'].iloc[0])

    h5_row = xcorr_df[xcorr_df['horizon_days'] == 5]
    pearson_r_h5 = float(h5_row['pearson_r'].iloc[0]) if not h5_row.empty else float('nan')

    # Mean abs Pearson r across horizons for the WS-3a clause.
    mean_abs_r = float(xcorr_df['pearson_r'].abs().mean())

    if not np.isnan(median_argmax_lag) and median_argmax_lag > 3.0:
        verdict = 'WS-3c'
        rationale = (
            f'median argmax_lag at tau=0.3 is {median_argmax_lag:.2f} days, '
            f'exceeding the 3-day threshold. BEAR_score leads argmax by a '
            f'meaningful margin -- argmax is suppressing a signal that is '
            f'already present.'
        )
    elif (not np.isnan(pearson_r_h5) and abs(pearson_r_h5) > 0.15
          and (np.isnan(median_argmax_lag) or median_argmax_lag <= 3.0)):
        verdict = 'WS-3a'
        rationale = (
            f'Pearson |r| at h=5d is {abs(pearson_r_h5):.3f} (>0.15) and '
            f'median argmax_lag at tau=0.3 is {median_argmax_lag:.2f} days '
            f'(<=3). BEAR_score is approximately coincident with the argmax '
            f'flip and carries real predictive content -- detector signal is '
            f'correct in timing but flickering between adjacent regimes '
            f'destroys usability.'
        )
    else:
        verdict = 'WS-3b'
        rationale = (
            f'Pearson |r| at h=5d is {abs(pearson_r_h5):.3f} (<=0.15) and '
            f'median argmax_lag at tau=0.3 is {median_argmax_lag:.2f}. '
            f'BEAR_score lacks both leading content and meaningful '
            f'forward-drawdown correlation -- the detector is fundamentally '
            f'late and no consumption-layer or smoothing fix can rescue it.'
        )

    lines = [
        '=== Experiment 3 Verdict ===',
        '',
        f'VERDICT: {verdict}',
        '',
        f'Rationale: {rationale}',
        '',
        '=== Supporting numbers ===',
        '',
        f'BEAR onsets (n): {n_onsets}',
        f'Mean gap_days (onset - trough): {gap_days:.2f}',
        '',
        'Threshold sweep -- median argmax_lag per tau:',
    ]
    for _, row in threshold_df.iterrows():
        lines.append(
            f'  tau={row["tau"]:.2f}  '
            f'median_argmax_lag={row["median_argmax_lag"]:.2f}  '
            f'mean_argmax_lag={row["mean_argmax_lag"]:.2f}  '
            f'median_trough_lag={row["median_trough_lag"]:.2f}  '
            f'n_events_crossed={int(row["n_events_crossed"])}'
        )
    lines.append('')
    lines.append('Cross-correlation -- Pearson r(BEAR_score, forward drawdown):')
    for _, row in xcorr_df.iterrows():
        lines.append(
            f'  h={int(row["horizon_days"]):2d}d  '
            f'pearson_r={row["pearson_r"]:+.4f}  '
            f'n_obs={int(row["n_obs"])}  '
            f'p_value={row["p_value"]:.2e}'
        )
    lines.append('')
    lines.append(f'Mean |Pearson r| across {len(xcorr_df)} horizons: {mean_abs_r:.4f}')
    return '\n'.join(lines) + '\n'


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = load_scores()
    panel = load_panel()  # currently unused beyond load logging -- kept for parity
    _ = panel

    onsets = identify_bear_onsets(scores)
    if not onsets:
        raise RuntimeError('No BEAR onsets found -- cannot proceed.')

    logger.info('[+] Analysis (a) event-study')
    per_event_df, summary_df = compute_event_study(scores, onsets)
    per_event_path = OUT_DIR / 'event_study_bear_score.csv'
    summary_path = OUT_DIR / 'event_study_summary.csv'
    per_event_df.to_csv(per_event_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f'    wrote {per_event_path} ({len(per_event_df)} rows)')
    logger.info(f'    wrote {summary_path} ({len(summary_df)} rows)')

    logger.info('[+] Analysis (b) cross-correlation')
    xcorr_df = compute_cross_correlation(scores)
    xcorr_path = OUT_DIR / 'cross_correlation.csv'
    xcorr_df.to_csv(xcorr_path, index=False)
    logger.info(f'    wrote {xcorr_path}')
    for _, row in xcorr_df.iterrows():
        logger.info(
            f'    h={int(row["horizon_days"]):2d}d  '
            f'r={row["pearson_r"]:+.4f}  p={row["p_value"]:.2e}  '
            f'n={int(row["n_obs"])}'
        )

    logger.info('[+] Analysis (c) threshold sweep')
    threshold_df = compute_threshold_sweep(scores, onsets)
    threshold_path = OUT_DIR / 'threshold_sweep.csv'
    threshold_df.to_csv(threshold_path, index=False)
    logger.info(f'    wrote {threshold_path}')
    for _, row in threshold_df.iterrows():
        logger.info(
            f'    tau={row["tau"]:.2f}  '
            f'median_argmax_lag={row["median_argmax_lag"]:.2f}  '
            f'median_trough_lag={row["median_trough_lag"]:.2f}  '
            f'n_crossed={int(row["n_events_crossed"])}'
        )

    gap_days = compute_onset_to_trough_gap(scores, onsets)
    logger.info(f'[+] Mean gap_days (onset - trough): {gap_days:.2f}')

    verdict_text = render_verdict(threshold_df, xcorr_df, gap_days, len(onsets))
    verdict_path = OUT_DIR / 'verdict.txt'
    verdict_path.write_text(verdict_text, encoding='utf-8')
    logger.info(f'[+] wrote {verdict_path}')
    for line in verdict_text.splitlines():
        logger.info(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
