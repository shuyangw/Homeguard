"""Experiment 8 -- V14 Action Convergence Diagnostic.

Post-hoc decomposition of the V14 factorial readiness output (commit 6f55e37):
V14a-cash / V14b-spy / V14c-dampen converged within 0.011 Sharpe at 5 bps
near_close. This diagnostic disambiguates WHICH mechanism is operative so that
the WS-3 detector-intervention spec can target the actual constraint.

Three hypotheses with falsifiable predictions (see spec for full text):

  M1 (rare-events ceiling):
       BEAR-soft total days < 5% of gated window
       OR median event duration < 5 trading days
       OR BEAR-soft partition contributes < 30% of V14-V11 excess

  M2 (action equivalence during real drawdowns):
       Cross-variant per-event correlation > 0.85
       AND pooled corr(SPY return, V11-plan return) during BEAR-soft > 0.85

  M3 (exit-timing failure):
       Median exit-to-SPY-low lag > 5 trading days
       AND mean 10-day post-exit V14a-V11 excess return < 0

Decision matrix maps (M1, M2, M3) -> WS-3 track (a / b / c.1 / d).

Inputs (existing artifacts; no new gates fire):
  diagnostics/regime/v0_scores/labels.parquet     (BEAR_score series for A6 sanity check)
  diagnostics/data/spy_vix_2016_2026.parquet      (SPY OHLC for A4/A5)
  config/research/v14_tau_constants.json          (tau_in=0.555556, tau_out=0.455556)

Run:
    PYTHONPATH=. python notebooks/research/experiment8_v14_action_convergence.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.research.ramp_phase4.config import HarnessConfig, load_v14_tau_constants
from src.research.ramp_phase4.engine import DailyRecord, run_variant
from src.research.ramp_phase4.metrics import sharpe_ratio
from src.research.ramp_phase4.variants import REGISTRY
from src.utils.logger import logger


OUT_DIR = Path('diagnostics/v14_action_convergence')
PANEL_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
SCORES_PATH = Path('diagnostics/regime/v0_scores/labels.parquet')
UNIVERSE_CSV = Path('config/universes/sp500-2025.csv')

START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2026, 5, 16)
INITIAL_CAPITAL = 100000.0
COST_BPS = 5.0
TIMING_MODE = 'near_close'
DELTA_REBALANCE_PCT = 0.02

VARIANTS = ('V11', 'V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen')
TRADING_DAYS_PER_YEAR = 252

# A6 counterfactual tau_out values.
A6_TAU_OUT_GRID = (0.20, 0.30, 0.40, 0.50)

# A5 exit-timing window.
EXIT_WINDOW = 20
POST_EXIT_HORIZONS = (5, 10, 20)


@dataclass
class VariantRun:
    label: str
    records: List[DailyRecord]
    sharpe: float


# ---------- Setup ----------

def build_cfg(soft_bear_tau_in: float, soft_bear_tau_out: float) -> HarnessConfig:
    return HarnessConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        universe_csv=UNIVERSE_CSV,
        initial_capital=INITIAL_CAPITAL,
        cost_bps_per_side=COST_BPS,
        timing_mode=TIMING_MODE,
        delta_rebalance_pct=DELTA_REBALANCE_PCT,
        soft_bear_tau_in=soft_bear_tau_in,
        soft_bear_tau_out=soft_bear_tau_out,
    )


def run_one(label: str, tau_in: float, tau_out: float) -> VariantRun:
    cfg = build_cfg(tau_in, tau_out)
    spec = REGISTRY[label]
    logger.info(f'[+] Running {label} @ 5bps near_close (tau_in={tau_in:.4f}, tau_out={tau_out:.4f})...')
    t0 = time.time()
    records = run_variant(cfg, spec)
    elapsed = time.time() - t0
    if not records:
        raise RuntimeError(f'{label} produced no records')
    rets = pd.Series([r.daily_return for r in records])
    sr = sharpe_ratio(rets)
    logger.info(
        f'[+] {label} done in {elapsed:.1f}s: Sharpe={sr:.4f} n_days={len(records)}'
    )
    return VariantRun(label=label, records=records, sharpe=sr)


def persist_records(run: VariantRun, path: Path) -> None:
    rows = []
    for r in run.records:
        rows.append({
            'date': r.date,
            'regime': r.regime,
            'daily_return': r.daily_return,
            'portfolio_value': r.portfolio_value,
            'turnover_usd': r.turnover_usd,
            'cost_usd': r.cost_usd,
            'target_weights_json': json.dumps(r.target_weights),
            'realized_weights_json': json.dumps(r.realized_weights),
        })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(path, index=False)
    logger.info(f'[+] Wrote {path} ({len(df)} rows)')


def build_returns_panel(runs: Dict[str, VariantRun]) -> pd.DataFrame:
    """Returns DataFrame indexed by date with columns daily_return per variant + regime + in_bear_soft_mode."""
    base = pd.DataFrame({
        'date': [r.date for r in runs['V14a-soft-bear-cash'].records],
        'regime_v14a': [r.regime for r in runs['V14a-soft-bear-cash'].records],
    })
    base['date'] = pd.to_datetime(base['date'])
    for label, run in runs.items():
        s = pd.Series([r.daily_return for r in run.records],
                      index=pd.to_datetime([r.date for r in run.records]),
                      name=label)
        base = base.set_index('date')
        base[label] = s
        base = base.reset_index()
    base = base.set_index('date').sort_index()
    base['in_bear_soft_mode'] = base['regime_v14a'].eq('BEAR_SOFT_CASH')
    return base


# ---------- A1: Coverage statistics ----------

def find_events(mask: pd.Series) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_inclusive) for contiguous True runs in mask."""
    arr = mask.to_numpy().astype(bool)
    events: List[Tuple[int, int]] = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            j = i
            while j + 1 < n and arr[j + 1]:
                j += 1
            events.append((i, j))
            i = j + 1
        else:
            i += 1
    return events


def a1_coverage(panel: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Build coverage table + return summary metrics for verdict synthesis."""
    mask = panel['in_bear_soft_mode']
    events = find_events(mask)
    durations = [e[1] - e[0] + 1 for e in events]
    total_days = int(mask.sum())
    gated_total = int(len(mask))
    pct_gated = 100.0 * total_days / gated_total if gated_total else 0.0

    # Per-year breakdown.
    by_year_days = panel.groupby(panel.index.year)['in_bear_soft_mode'].sum()
    by_year_events: Dict[int, int] = {}
    for s, e in events:
        yr = int(panel.index[s].year)
        by_year_events[yr] = by_year_events.get(yr, 0) + 1

    rows = []
    rows.append({'metric': 'total_bear_soft_days', 'value': total_days})
    rows.append({'metric': 'gated_window_days', 'value': gated_total})
    rows.append({'metric': 'bear_soft_pct_of_gated', 'value': round(pct_gated, 4)})
    rows.append({'metric': 'n_events', 'value': len(events)})
    if durations:
        rows.append({'metric': 'event_duration_median', 'value': float(np.median(durations))})
        rows.append({'metric': 'event_duration_p25', 'value': float(np.percentile(durations, 25))})
        rows.append({'metric': 'event_duration_p75', 'value': float(np.percentile(durations, 75))})
        rows.append({'metric': 'event_duration_max', 'value': float(np.max(durations))})
        rows.append({'metric': 'event_duration_mean', 'value': float(np.mean(durations))})
    else:
        for k in ('event_duration_median', 'event_duration_p25', 'event_duration_p75',
                  'event_duration_max', 'event_duration_mean'):
            rows.append({'metric': k, 'value': float('nan')})
    for yr, days in sorted(by_year_days.items()):
        rows.append({'metric': f'year_{yr}_bear_soft_days', 'value': int(days)})
    for yr, n in sorted(by_year_events.items()):
        rows.append({'metric': f'year_{yr}_n_events', 'value': int(n)})
    df = pd.DataFrame(rows)
    metrics = {
        'total_bear_soft_days': total_days,
        'gated_window_days': gated_total,
        'pct_of_gated': pct_gated,
        'n_events': len(events),
        'median_duration': float(np.median(durations)) if durations else float('nan'),
    }
    return df, metrics


# ---------- A2: Action-attribution decomposition ----------

def a2_attribution(panel: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Per variant: partition daily_excess by BEAR-soft mode and compute summary stats."""
    v11 = panel['V11']
    mask = panel['in_bear_soft_mode']
    out_rows = []
    per_variant: Dict[str, Dict[str, float]] = {}
    for variant in ('V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen'):
        excess = panel[variant] - v11
        for partition_name, partition_mask in (('BEAR_SOFT', mask), ('NOT_BEAR_SOFT', ~mask)):
            seg = excess[partition_mask]
            n = int(partition_mask.sum())
            mean = float(seg.mean()) if n > 0 else 0.0
            std = float(seg.std(ddof=1)) if n > 1 else 0.0
            sum_excess = float(seg.sum()) if n > 0 else 0.0
            sharpe_contrib = (
                (mean * TRADING_DAYS_PER_YEAR) / (std * np.sqrt(TRADING_DAYS_PER_YEAR))
                if std > 1e-15 else 0.0
            )
            out_rows.append({
                'variant': variant,
                'partition': partition_name,
                'n_days': n,
                'sum_excess': sum_excess,
                'mean_excess': mean,
                'std_excess': std,
                'partition_sharpe_annual': sharpe_contrib,
            })
        bear_seg = excess[mask]
        nb_seg = excess[~mask]
        bear_sum = float(bear_seg.sum()) if len(bear_seg) else 0.0
        total_sum = float(excess.sum())
        bear_pct = (bear_sum / total_sum * 100.0) if abs(total_sum) > 1e-15 else float('nan')
        bear_std = float(bear_seg.std(ddof=1)) if len(bear_seg) > 1 else float('nan')
        out_rows.append({
            'variant': variant,
            'partition': 'BEAR_SOFT_share_of_total_excess_pct',
            'n_days': int(mask.sum()),
            'sum_excess': bear_sum,
            'mean_excess': float('nan'),
            'std_excess': bear_std,
            'partition_sharpe_annual': bear_pct,
        })
        per_variant[variant] = {
            'bear_soft_sum_excess': bear_sum,
            'bear_soft_pct_of_total_excess': bear_pct,
            'bear_soft_std_excess': bear_std,
            'nb_std_excess': float(nb_seg.std(ddof=1)) if len(nb_seg) > 1 else float('nan'),
            'total_excess': total_sum,
        }
    return pd.DataFrame(out_rows), per_variant


# ---------- A3: Per-event P&L by variant ----------

def cumulative_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((1 + returns).prod() - 1.0)


def a3_per_event(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    mask = panel['in_bear_soft_mode']
    events = find_events(mask)
    rows = []
    cum_v14a: List[float] = []
    cum_v14b: List[float] = []
    cum_v14c: List[float] = []
    for idx, (s, e) in enumerate(events, start=1):
        window = panel.iloc[s:e + 1]
        ev = {
            'event_id': idx,
            'start_date': window.index[0].date().isoformat(),
            'end_date': window.index[-1].date().isoformat(),
            'duration_days': e - s + 1,
            'cum_v11': cumulative_return(window['V11']),
            'cum_v14a': cumulative_return(window['V14a-soft-bear-cash']),
            'cum_v14b': cumulative_return(window['V14b-soft-bear-spy']),
            'cum_v14c': cumulative_return(window['V14c-soft-bear-dampen']),
        }
        ev['v14a_minus_v11'] = ev['cum_v14a'] - ev['cum_v11']
        ev['v14b_minus_v11'] = ev['cum_v14b'] - ev['cum_v11']
        ev['v14c_minus_v11'] = ev['cum_v14c'] - ev['cum_v11']
        rows.append(ev)
        cum_v14a.append(ev['cum_v14a'])
        cum_v14b.append(ev['cum_v14b'])
        cum_v14c.append(ev['cum_v14c'])
    df_events = pd.DataFrame(rows)
    corr_mat = pd.DataFrame(index=['V14a', 'V14b', 'V14c'], columns=['V14a', 'V14b', 'V14c'], dtype=float)
    arrs = {'V14a': np.array(cum_v14a), 'V14b': np.array(cum_v14b), 'V14c': np.array(cum_v14c)}
    for a in arrs:
        for b in arrs:
            if len(arrs[a]) >= 2 and len(arrs[b]) >= 2:
                corr_mat.loc[a, b] = float(np.corrcoef(arrs[a], arrs[b])[0, 1])
            else:
                corr_mat.loc[a, b] = float('nan')
    summary = {
        'corr_v14a_v14b': float(corr_mat.loc['V14a', 'V14b']),
        'corr_v14a_v14c': float(corr_mat.loc['V14a', 'V14c']),
        'corr_v14b_v14c': float(corr_mat.loc['V14b', 'V14c']),
        'n_events': len(events),
    }
    return df_events, corr_mat, summary


# ---------- A4: SPY vs V11-plan return correlation ----------

def load_spy_returns() -> pd.Series:
    panel = pd.read_parquet(PANEL_PATH)
    spy = panel['spy_close'].pct_change().dropna()
    return spy


def a4_spy_v11_corr(panel: pd.DataFrame, spy_returns: pd.Series) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mask = panel['in_bear_soft_mode']
    events = find_events(mask)
    rows = []
    pooled_spy: List[float] = []
    pooled_v11: List[float] = []
    for idx, (s, e) in enumerate(events, start=1):
        window = panel.iloc[s:e + 1]
        spy_window = spy_returns.reindex(window.index)
        v11_window = window['V11']
        merged = pd.DataFrame({'spy': spy_window, 'v11': v11_window}).dropna()
        per_corr = float('nan')
        if len(merged) >= 2 and merged['spy'].std() > 1e-15 and merged['v11'].std() > 1e-15:
            per_corr = float(np.corrcoef(merged['spy'], merged['v11'])[0, 1])
        pooled_spy.extend(merged['spy'].tolist())
        pooled_v11.extend(merged['v11'].tolist())
        rows.append({
            'event_id': idx,
            'start_date': window.index[0].date().isoformat(),
            'end_date': window.index[-1].date().isoformat(),
            'duration_days': e - s + 1,
            'n_used': len(merged),
            'corr_spy_v11': per_corr,
        })
    pooled_arr_spy = np.array(pooled_spy, dtype=float)
    pooled_arr_v11 = np.array(pooled_v11, dtype=float)
    pooled_corr = float('nan')
    if len(pooled_arr_spy) >= 2 and pooled_arr_spy.std() > 1e-15 and pooled_arr_v11.std() > 1e-15:
        pooled_corr = float(np.corrcoef(pooled_arr_spy, pooled_arr_v11)[0, 1])
    rows.append({
        'event_id': 'POOLED',
        'start_date': '',
        'end_date': '',
        'duration_days': len(pooled_arr_spy),
        'n_used': len(pooled_arr_spy),
        'corr_spy_v11': pooled_corr,
    })
    summary = {
        'pooled_corr_spy_v11': pooled_corr,
        'n_pooled_days': len(pooled_arr_spy),
    }
    return pd.DataFrame(rows), summary


# ---------- A5: Exit-timing analysis ----------

def a5_exit_timing(panel: pd.DataFrame, spy_close: pd.Series) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """For each True -> False transition, find SPY local min within +/-20d and post-exit excess."""
    mask = panel['in_bear_soft_mode']
    arr = mask.to_numpy().astype(bool)
    exit_indices: List[int] = []
    for i in range(1, len(arr)):
        if arr[i - 1] and not arr[i]:
            exit_indices.append(i)  # i is the first non-BEAR_SOFT day (the day after exit)

    # Align SPY closes to panel index.
    spy_aligned = spy_close.reindex(panel.index).ffill()

    excess_5d: List[float] = []
    excess_10d: List[float] = []
    excess_20d: List[float] = []
    lags: List[int] = []
    rows = []
    for idx, ei in enumerate(exit_indices, start=1):
        exit_date = panel.index[ei]
        lo = max(0, ei - EXIT_WINDOW)
        hi = min(len(panel) - 1, ei + EXIT_WINDOW)
        spy_slice = spy_aligned.iloc[lo:hi + 1].to_numpy()
        if np.all(np.isnan(spy_slice)):
            continue
        # argmin over the window (NaN-safe).
        valid_min = np.nanmin(spy_slice)
        min_positions = np.where(spy_slice == valid_min)[0]
        if len(min_positions) == 0:
            continue
        min_pos_in_window = int(min_positions[0])
        min_idx_global = lo + min_pos_in_window
        lag = int(min_idx_global - ei)
        lags.append(lag)

        # Post-exit V14a - V11 cumulative excess.
        excess_h: Dict[int, float] = {}
        for h in POST_EXIT_HORIZONS:
            end_h = min(len(panel) - 1, ei + h - 1)
            if end_h < ei:
                excess_h[h] = float('nan')
                continue
            window = panel.iloc[ei:end_h + 1]
            v14a_cum = cumulative_return(window['V14a-soft-bear-cash'])
            v11_cum = cumulative_return(window['V11'])
            excess_h[h] = float(v14a_cum - v11_cum)
        excess_5d.append(excess_h[5])
        excess_10d.append(excess_h[10])
        excess_20d.append(excess_h[20])

        rows.append({
            'exit_id': idx,
            'exit_date': exit_date.date().isoformat(),
            'spy_low_date': panel.index[min_idx_global].date().isoformat(),
            'lag_days': lag,
            'spy_close_at_exit': float(spy_aligned.iloc[ei]) if not np.isnan(spy_aligned.iloc[ei]) else float('nan'),
            'spy_low_value': float(valid_min),
            'excess_v14a_minus_v11_5d': excess_h[5],
            'excess_v14a_minus_v11_10d': excess_h[10],
            'excess_v14a_minus_v11_20d': excess_h[20],
        })
    df = pd.DataFrame(rows)
    median_lag = float(np.median(lags)) if lags else float('nan')
    mean_10d = float(np.nanmean(excess_10d)) if excess_10d else float('nan')
    summary = {
        'n_exits': len(exit_indices),
        'median_lag_days': median_lag,
        'mean_lag_days': float(np.mean(lags)) if lags else float('nan'),
        'mean_5d_excess': float(np.nanmean(excess_5d)) if excess_5d else float('nan'),
        'mean_10d_excess': mean_10d,
        'mean_20d_excess': float(np.nanmean(excess_20d)) if excess_20d else float('nan'),
    }
    return df, summary


# ---------- A6: Counterfactual tau_out sweep ----------

def load_bear_score_series() -> pd.Series:
    scores = pd.read_parquet(SCORES_PATH)
    scores['date'] = pd.to_datetime(scores['date'])
    s = scores.set_index('date')['score_BEAR']
    return s


def schmitt_mode(bear_score: pd.Series, tau_in: float, tau_out: float) -> pd.Series:
    """Reconstruct in_bear_soft_mode via Schmitt trigger on the BEAR_score series."""
    arr = bear_score.to_numpy()
    mode = np.zeros(len(arr), dtype=bool)
    current = False
    for i, x in enumerate(arr):
        if not np.isnan(x):
            if not current and x >= tau_in:
                current = True
            elif current and x < tau_out:
                current = False
        mode[i] = current
    return pd.Series(mode, index=bear_score.index, name='in_bear_soft_mode')


def a6_tau_out_sweep(
    panel: pd.DataFrame,
    bear_score: pd.Series,
    tau_in: float,
    actual_tau_out: float,
) -> pd.DataFrame:
    """Counterfactual hypothetical V14a Sharpe at alternate tau_out values.

    Approximation:
      - on days that were originally in_bear_soft_mode AND are in the new mode
        -> use V14a's actual return (= 0 for cash, but engine has costs etc.;
           we use the recorded daily_return)
      - on days NOT in original mode but IN new mode
        -> V14a would have been cash; hypothetical return = 0
      - on days in original mode but NOT in new mode
        -> V14a would have followed V11; use V11's actual daily_return
      - on days in neither: V14a == V11 by V14 construction; use V14a actual.

    The actual_tau_out (0.455556) reconstruction should approximately recover
    V14a's true Sharpe; we report it explicitly as a sanity row.
    """
    original_mode = panel['in_bear_soft_mode']
    # Align bear_score to panel.
    aligned_score = bear_score.reindex(panel.index).ffill()
    rows = []
    # Include actual tau_out as a sanity row.
    grid = list(A6_TAU_OUT_GRID) + [actual_tau_out]
    grid = sorted(set(round(x, 6) for x in grid))
    for tau_out in grid:
        new_mode = schmitt_mode(aligned_score, tau_in, tau_out)
        new_mode = new_mode.reindex(panel.index).fillna(False)
        hyp_returns = []
        for i, ts in enumerate(panel.index):
            in_orig = bool(original_mode.iloc[i])
            in_new = bool(new_mode.iloc[i])
            if in_orig and in_new:
                hyp_returns.append(panel['V14a-soft-bear-cash'].iloc[i])
            elif (not in_orig) and in_new:
                hyp_returns.append(0.0)
            elif in_orig and (not in_new):
                hyp_returns.append(panel['V11'].iloc[i])
            else:
                hyp_returns.append(panel['V14a-soft-bear-cash'].iloc[i])
        hyp_series = pd.Series(hyp_returns, index=panel.index)
        hyp_sharpe = sharpe_ratio(hyp_series)
        rows.append({
            'tau_out': tau_out,
            'tau_in': tau_in,
            'is_actual_tau_out': abs(tau_out - actual_tau_out) < 1e-4,
            'n_bear_soft_days_new': int(new_mode.sum()),
            'n_bear_soft_days_original': int(original_mode.sum()),
            'hypothetical_v14a_sharpe': float(hyp_sharpe),
        })
    return pd.DataFrame(rows)


# ---------- Verdict synthesis ----------

def synthesize_verdict(
    a1_metrics: Dict[str, float],
    a2_per_variant: Dict[str, Dict[str, float]],
    a3_summary: Dict[str, float],
    a4_summary: Dict[str, float],
    a5_summary: Dict[str, float],
) -> Tuple[str, Dict[str, str], str]:
    """Return (verdict_text, verdicts_dict, recommended_track)."""
    # M1: rare-events ceiling.
    # support if total < 5% OR median duration < 5 OR BEAR-soft contributes <30% of excess (per variant).
    pct_gated = a1_metrics['pct_of_gated']
    median_dur = a1_metrics['median_duration']
    v14a_bear_share = a2_per_variant['V14a-soft-bear-cash']['bear_soft_pct_of_total_excess']
    m1_total_days_small = pct_gated < 5.0
    m1_short_events = not np.isnan(median_dur) and median_dur < 5.0
    m1_low_share = not np.isnan(v14a_bear_share) and abs(v14a_bear_share) < 30.0
    m1_signals = [m1_total_days_small, m1_short_events, m1_low_share]
    if sum(1 for s in m1_signals if s) >= 2:
        m1 = 'supported'
    elif sum(1 for s in m1_signals if s) == 0:
        m1 = 'refuted'
    else:
        m1 = 'inconclusive'

    # M2: action equivalence.
    # support if cross-variant per-event correlation > 0.85 AND pooled corr(SPY, V11) > 0.85.
    cross_corrs = [
        a3_summary['corr_v14a_v14b'],
        a3_summary['corr_v14a_v14c'],
        a3_summary['corr_v14b_v14c'],
    ]
    cross_corrs_valid = [c for c in cross_corrs if not np.isnan(c)]
    min_cross = float(min(cross_corrs_valid)) if cross_corrs_valid else float('nan')
    pooled = a4_summary['pooled_corr_spy_v11']
    m2_cross_pass = not np.isnan(min_cross) and min_cross > 0.85
    m2_pooled_pass = not np.isnan(pooled) and pooled > 0.85
    if m2_cross_pass and m2_pooled_pass:
        m2 = 'supported'
    elif (not m2_cross_pass) and (not m2_pooled_pass):
        m2 = 'refuted'
    else:
        m2 = 'inconclusive'

    # M3: exit-timing failure.
    # support if median lag > 5 AND mean 10d post-exit excess < 0.
    median_lag = a5_summary['median_lag_days']
    mean_10d = a5_summary['mean_10d_excess']
    m3_lag_pass = not np.isnan(median_lag) and median_lag > 5.0
    m3_excess_pass = not np.isnan(mean_10d) and mean_10d < 0.0
    if m3_lag_pass and m3_excess_pass:
        m3 = 'supported'
    elif (not m3_lag_pass) and (not m3_excess_pass):
        m3 = 'refuted'
    else:
        m3 = 'inconclusive'

    verdicts = {'M1': m1, 'M2': m2, 'M3': m3}

    # Decision matrix per spec.
    def vstr(v: str) -> str:
        return v

    sup = lambda v: verdicts[v] == 'supported'
    ref = lambda v: verdicts[v] == 'refuted'
    inc = lambda v: verdicts[v] == 'inconclusive'

    if sup('M1') and ref('M2') and ref('M3'):
        track = 'WS-3b (leading indicators)'
    elif ref('M1') and sup('M2') and ref('M3'):
        track = 'WS-3a (detector hysteresis)'
    elif ref('M1') and ref('M2') and sup('M3'):
        track = 'WS-3c.1 (consumer exit logic)'
    elif sup('M1') and sup('M2'):
        track = 'WS-3d (detector replacement)'
    elif sup('M2') and sup('M3'):
        track = 'WS-3a + WS-3c.1 in parallel'
    elif sup('M1') and sup('M3'):
        track = 'WS-3b primary, WS-3c.1 fallback'
    elif sup('M1') and sup('M2') and sup('M3'):
        track = 'WS-3d (incremental fixes will not compound enough)'
    elif (not sup('M1')) and (not sup('M2')) and (not sup('M3')):
        track = 'WS-3d with expanded scope OR halt WS-3 and pursue alternative strategies'
    else:
        # Catch-all for partial / inconclusive: prefer the strongest single supported mechanism.
        supports = [v for v in ('M1', 'M2', 'M3') if sup(v)]
        if len(supports) == 1:
            track_map = {
                'M1': 'WS-3b (leading indicators)',
                'M2': 'WS-3a (detector hysteresis)',
                'M3': 'WS-3c.1 (consumer exit logic)',
            }
            track = track_map[supports[0]] + ' (other mechanisms inconclusive; fallback caveat)'
        else:
            track = 'WS-3d (mechanisms inconclusive; broadest intervention)'

    lines = []
    lines.append('=== Experiment 8 Verdict ===')
    lines.append('')
    lines.append(f'M1 (rare events): {m1}')
    lines.append(f'  - Total BEAR-soft days: {a1_metrics["total_bear_soft_days"]} '
                 f'({pct_gated:.2f}% of gated window)')
    lines.append(f'  - Median event duration: {median_dur:.1f} days')
    lines.append(f'  - BEAR-soft partition contribution to V14a-V11 excess: {v14a_bear_share:.2f}%')
    lines.append('')
    lines.append(f'M2 (action equivalence): {m2}')
    lines.append(f'  - Cross-variant per-event correlation: '
                 f'{a3_summary["corr_v14a_v14b"]:.4f} (V14a/V14b), '
                 f'{a3_summary["corr_v14a_v14c"]:.4f} (V14a/V14c), '
                 f'{a3_summary["corr_v14b_v14c"]:.4f} (V14b/V14c)')
    lines.append(f'  - Pooled corr(SPY, V11) during BEAR-soft: {pooled:.4f} '
                 f'(n={a4_summary["n_pooled_days"]} days)')
    lines.append('')
    lines.append(f'M3 (exit-timing failure): {m3}')
    lines.append(f'  - Median exit-to-SPY-low lag: {median_lag:.1f} days')
    lines.append(f'  - Mean 10d post-exit V14a-V11 excess: {mean_10d*100:.4f}%')
    lines.append(f'  - N exits analyzed: {a5_summary["n_exits"]}')
    lines.append('')
    lines.append(f'Decision matrix output: {track}')
    lines.append('')
    lines.append('Interpretation: With M1 {m1}, M2 {m2}, M3 {m3}, the diagnostic points '
                 'to {track} as the next intervention. The V14a/b/c convergence is '
                 'consistent with this mechanism reading; the verdict supersedes assumed '
                 'priors about which axis (trigger / hysteresis / exit logic) is binding.'
                 .format(m1=m1, m2=m2, m3=m3, track=track))
    return '\n'.join(lines), verdicts, track


# ---------- One_day_lag sanity check ----------

def lag_sanity_check() -> Dict[str, object]:
    """Run V11 + V14a at 5 bps one_day_lag and recompute A1 (coverage), A4 (corr), A5 (lag)."""
    logger.info('[+] Sanity check: re-running V11 + V14a at one_day_lag for A1/A4/A5...')
    tau_in, tau_out = load_v14_tau_constants()

    def _build_lag_cfg() -> HarnessConfig:
        return HarnessConfig(
            start_date=START_DATE,
            end_date=END_DATE,
            universe_csv=UNIVERSE_CSV,
            initial_capital=INITIAL_CAPITAL,
            cost_bps_per_side=COST_BPS,
            timing_mode='one_day_lag',
            delta_rebalance_pct=DELTA_REBALANCE_PCT,
            soft_bear_tau_in=tau_in,
            soft_bear_tau_out=tau_out,
        )

    runs = {}
    for label in ('V11', 'V14a-soft-bear-cash'):
        cfg = _build_lag_cfg()
        t0 = time.time()
        recs = run_variant(cfg, REGISTRY[label])
        logger.info(f'[+] {label} (one_day_lag) done in {time.time()-t0:.1f}s')
        runs[label] = recs
    # Build a panel.
    dates = [r.date for r in runs['V14a-soft-bear-cash']]
    df = pd.DataFrame({'date': pd.to_datetime(dates)})
    df['regime_v14a'] = [r.regime for r in runs['V14a-soft-bear-cash']]
    df['V11'] = [r.daily_return for r in runs['V11']]
    df['V14a-soft-bear-cash'] = [r.daily_return for r in runs['V14a-soft-bear-cash']]
    df = df.set_index('date').sort_index()
    df['in_bear_soft_mode'] = df['regime_v14a'].eq('BEAR_SOFT_CASH')

    mask = df['in_bear_soft_mode']
    events = find_events(mask)
    durations = [e[1] - e[0] + 1 for e in events]
    spy_close = pd.read_parquet(PANEL_PATH)['spy_close']
    spy_returns = spy_close.pct_change().dropna()

    # A4: pooled corr.
    pooled_spy: List[float] = []
    pooled_v11: List[float] = []
    for s, e in events:
        window = df.iloc[s:e + 1]
        spy_w = spy_returns.reindex(window.index)
        merged = pd.DataFrame({'spy': spy_w, 'v11': window['V11']}).dropna()
        pooled_spy.extend(merged['spy'].tolist())
        pooled_v11.extend(merged['v11'].tolist())
    pooled_corr = float('nan')
    if len(pooled_spy) >= 2:
        arr_spy = np.array(pooled_spy)
        arr_v11 = np.array(pooled_v11)
        if arr_spy.std() > 1e-15 and arr_v11.std() > 1e-15:
            pooled_corr = float(np.corrcoef(arr_spy, arr_v11)[0, 1])

    # A5: exit-timing lag.
    arr = mask.to_numpy().astype(bool)
    exit_indices: List[int] = []
    for i in range(1, len(arr)):
        if arr[i - 1] and not arr[i]:
            exit_indices.append(i)
    spy_aligned = spy_close.reindex(df.index).ffill()
    lags: List[int] = []
    excess_10d: List[float] = []
    for ei in exit_indices:
        lo = max(0, ei - EXIT_WINDOW)
        hi = min(len(df) - 1, ei + EXIT_WINDOW)
        spy_slice = spy_aligned.iloc[lo:hi + 1].to_numpy()
        if np.all(np.isnan(spy_slice)):
            continue
        valid_min = np.nanmin(spy_slice)
        positions = np.where(spy_slice == valid_min)[0]
        if len(positions) == 0:
            continue
        min_idx_global = lo + int(positions[0])
        lags.append(min_idx_global - ei)
        end_h = min(len(df) - 1, ei + 10 - 1)
        if end_h >= ei:
            window = df.iloc[ei:end_h + 1]
            v14a_cum = cumulative_return(window['V14a-soft-bear-cash'])
            v11_cum = cumulative_return(window['V11'])
            excess_10d.append(float(v14a_cum - v11_cum))

    return {
        'mode': 'one_day_lag',
        'n_bear_soft_days': int(mask.sum()),
        'pct_gated': 100.0 * mask.sum() / len(mask),
        'n_events': len(events),
        'median_duration': float(np.median(durations)) if durations else float('nan'),
        'pooled_corr_spy_v11': pooled_corr,
        'n_exits': len(exit_indices),
        'median_lag_days': float(np.median(lags)) if lags else float('nan'),
        'mean_10d_excess': float(np.mean(excess_10d)) if excess_10d else float('nan'),
    }


# ---------- Main ----------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tau_in, tau_out = load_v14_tau_constants()
    logger.info(f'[+] Loaded tau_in={tau_in:.6f} tau_out={tau_out:.6f}')

    t_start = time.time()

    # 1. Run 4 backtests.
    runs: Dict[str, VariantRun] = {}
    for label in VARIANTS:
        runs[label] = run_one(label, tau_in, tau_out)

    # 2. Persist records.
    parquet_map = {
        'V11': 'v11_records.parquet',
        'V14a-soft-bear-cash': 'v14a_records.parquet',
        'V14b-soft-bear-spy': 'v14b_records.parquet',
        'V14c-soft-bear-dampen': 'v14c_records.parquet',
    }
    for label, fname in parquet_map.items():
        persist_records(runs[label], OUT_DIR / fname)

    # 3. Build aligned panel.
    panel = build_returns_panel(runs)
    logger.info(f'[+] Built panel: {len(panel)} days, {int(panel["in_bear_soft_mode"].sum())} '
                f'BEAR-soft days')

    # Independent sanity check: reconstruct in_bear_soft_mode from BEAR_score.
    bear_score = load_bear_score_series()
    reconstructed = schmitt_mode(bear_score.reindex(panel.index).ffill(), tau_in, tau_out)
    overlap = (panel['in_bear_soft_mode'].to_numpy() == reconstructed.to_numpy()).mean() * 100.0
    n_engine = int(panel['in_bear_soft_mode'].sum())
    n_recon = int(reconstructed.sum())
    logger.info(f'[+] Schmitt reconstruction overlap with engine: {overlap:.2f}% '
                f'(engine={n_engine} days, reconstructed={n_recon} days)')

    # 4. A1 -- coverage.
    a1_df, a1_metrics = a1_coverage(panel)
    a1_df.to_csv(OUT_DIR / 'a1_coverage.csv', index=False)
    logger.info(f'[+] A1: total BEAR-soft days={a1_metrics["total_bear_soft_days"]} '
                f'({a1_metrics["pct_of_gated"]:.2f}% of gated), n_events={a1_metrics["n_events"]}, '
                f'median duration={a1_metrics["median_duration"]:.1f}')

    # 5. A2 -- attribution.
    a2_df, a2_per_variant = a2_attribution(panel)
    a2_df.to_csv(OUT_DIR / 'a2_attribution.csv', index=False)
    for variant, info in a2_per_variant.items():
        logger.info(f'[+] A2 {variant}: BEAR-soft share={info["bear_soft_pct_of_total_excess"]:.2f}% '
                    f'of total excess, std_bear={info["bear_soft_std_excess"]:.6f}, '
                    f'std_nb={info["nb_std_excess"]:.6f}')

    # 6. A3 -- per-event.
    a3_df, a3_corr, a3_summary = a3_per_event(panel)
    a3_df.to_csv(OUT_DIR / 'a3_per_event.csv', index=False)
    a3_corr.to_csv(OUT_DIR / 'a3_corr_matrix.csv')
    logger.info(f'[+] A3: V14a/V14b={a3_summary["corr_v14a_v14b"]:.4f}, '
                f'V14a/V14c={a3_summary["corr_v14a_v14c"]:.4f}, '
                f'V14b/V14c={a3_summary["corr_v14b_v14c"]:.4f} '
                f'(n_events={a3_summary["n_events"]})')

    # 7. A4 -- SPY vs V11 correlation.
    spy_returns = load_spy_returns()
    a4_df, a4_summary = a4_spy_v11_corr(panel, spy_returns)
    a4_df.to_csv(OUT_DIR / 'a4_spy_v11_corr.csv', index=False)
    logger.info(f'[+] A4: pooled corr(SPY, V11) during BEAR-soft={a4_summary["pooled_corr_spy_v11"]:.4f} '
                f'(n={a4_summary["n_pooled_days"]} days)')

    # 8. A5 -- exit timing.
    spy_close = pd.read_parquet(PANEL_PATH)['spy_close']
    a5_df, a5_summary = a5_exit_timing(panel, spy_close)
    a5_df.to_csv(OUT_DIR / 'a5_exit_timing.csv', index=False)
    logger.info(f'[+] A5: n_exits={a5_summary["n_exits"]}, '
                f'median lag={a5_summary["median_lag_days"]:.1f}d, '
                f'mean 10d excess={a5_summary["mean_10d_excess"]*100:.4f}%')

    # 9. A6 -- tau_out sweep.
    a6_df = a6_tau_out_sweep(panel, bear_score, tau_in, tau_out)
    a6_df.to_csv(OUT_DIR / 'a6_tau_out_sweep.csv', index=False)
    logger.info(f'[+] A6 tau_out sweep written ({len(a6_df)} rows)')

    # 10. Verdict.
    verdict_text, verdicts, track = synthesize_verdict(
        a1_metrics=a1_metrics,
        a2_per_variant=a2_per_variant,
        a3_summary=a3_summary,
        a4_summary=a4_summary,
        a5_summary=a5_summary,
    )
    (OUT_DIR / 'verdict.txt').write_text(verdict_text + '\n')
    logger.info('[+] Verdict written:')
    for line in verdict_text.split('\n'):
        logger.info(f'    {line}')

    # 11. One_day_lag sanity check (A1, A4, A5 only).
    lag_results = lag_sanity_check()
    lag_lines = ['', '--- one_day_lag sanity check ---']
    for k, v in lag_results.items():
        lag_lines.append(f'{k}: {v}')
    with open(OUT_DIR / 'verdict.txt', 'a') as f:
        f.write('\n'.join(lag_lines) + '\n')
    logger.info(f'[+] one_day_lag sanity: median_lag={lag_results["median_lag_days"]:.1f}d, '
                f'pooled_corr={lag_results["pooled_corr_spy_v11"]:.4f}, '
                f'mean_10d_excess={lag_results["mean_10d_excess"]*100:.4f}%')

    # Save summary metrics JSON for the report builder.
    summary = {
        'tau_in': tau_in,
        'tau_out': tau_out,
        'a1': a1_metrics,
        'a2': a2_per_variant,
        'a3': a3_summary,
        'a4': a4_summary,
        'a5': a5_summary,
        'verdicts': verdicts,
        'track': track,
        'schmitt_overlap_pct': overlap,
        'schmitt_engine_days': n_engine,
        'schmitt_reconstructed_days': n_recon,
        'lag_sanity_check': lag_results,
    }
    (OUT_DIR / '_summary.json').write_text(json.dumps(summary, indent=2, default=str))

    total = time.time() - t_start
    logger.info(f'[+] All analyses complete in {total/60:.2f} min')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
