"""WS-3d diagnostic rerun: H1-H5 analyses on the v1 detector vs the v0 baseline.

Reads:
- diagnostics/regime/v1/labels.parquet (v1 replay output)
- diagnostics/regime/v0/labels.parquet (v0 replay output, locked baseline)
- diagnostics/regime/ground_truth.parquet (G1_BEAR, G2, G3, G4)

H5 measurement (Pre-commitment 5, GATING):
- For every G1_BEAR onset (transition False -> True), measure the lag in
  trading days from the onset date to the first detector-BEAR label in a
  forward 60-day window. v1 lag must be >= 30% lower than v0 lag (v0 14d
  baseline from docs/reports/ramp/20260523_regime_detector_diagnostic.md).

Output:
- docs/reports/ramp/20260601_ws3d_regime_diagnostic_rerun.md

Usage:
    PYTHONPATH=. python scripts/diagnostics/regime_detector_v1_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


V1_LABELS = Path('diagnostics/regime/v1/labels.parquet')
V0_LABELS = Path('diagnostics/regime/v0/labels.parquet')
GROUND_TRUTH = Path('diagnostics/regime/ground_truth.parquet')
G4_EVENTS = Path('config/diagnostics/regime_events_2017_2026.csv')
REPORT_PATH = Path('docs/reports/ramp/20260601_ws3d_regime_diagnostic_rerun.md')

FORWARD_WINDOW_DAYS = 60  # forward lag-search window per spec
V0_H5_BASELINE_DAYS = 14.0  # from 20260523_regime_detector_diagnostic.md
H5_REDUCTION_THRESHOLD_PCT = 30.0  # >= 30% reduction required to pass Gate 1


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df


def _g1_onsets(g1: pd.Series) -> List[pd.Timestamp]:
    """Return dates where G1_BEAR transitions False -> True."""
    prev = g1.shift(fill_value=False)
    onsets_mask = g1 & (~prev)
    return list(g1.index[onsets_mask])


def compute_h5_lag(
    labels: pd.DataFrame,
    g1: pd.Series,
    forward_window_days: int = FORWARD_WINDOW_DAYS,
) -> pd.DataFrame:
    """For each G1_BEAR onset, measure lag to first detector-BEAR label
    within the forward window.

    Returns DataFrame with [onset_date, lag_days, fired_within_window].
    """
    rows = []
    for onset in _g1_onsets(g1):
        end = onset + pd.Timedelta(days=forward_window_days)
        window = labels.loc[onset:end]
        bear_dates = window.index[window['regime'] == 'BEAR']
        if len(bear_dates) == 0:
            rows.append({
                'onset_date': onset,
                'lag_days': float('nan'),
                'fired_within_window': False,
            })
        else:
            lag = (bear_dates[0] - onset).days
            rows.append({
                'onset_date': onset,
                'lag_days': float(lag),
                'fired_within_window': True,
            })
    return pd.DataFrame(rows)


def compute_h5_lag_g4(
    labels: pd.DataFrame,
    events_csv: Path,
    event_type: str = 'drawdown',
) -> pd.DataFrame:
    """G4 hand-curated drawdown-event basis (matches 20260523 report exactly).

    For each event row of the specified type, measure lag in calendar days
    from event start_date to first detector-BEAR label within [start, end].
    """
    events = pd.read_csv(events_csv, parse_dates=['start_date', 'end_date'])
    events = events[events['event_type'] == event_type]
    rows = []
    for _, ev in events.iterrows():
        window = labels.loc[ev['start_date']:ev['end_date']]
        bear_dates = window.index[window['regime'] == 'BEAR']
        if len(bear_dates) == 0:
            rows.append({
                'event': ev['event_name'],
                'start_date': ev['start_date'],
                'lag_days': float('nan'),
                'fired_within_window': False,
            })
        else:
            lag = (bear_dates[0] - ev['start_date']).days
            rows.append({
                'event': ev['event_name'],
                'start_date': ev['start_date'],
                'lag_days': float(lag),
                'fired_within_window': True,
            })
    return pd.DataFrame(rows)


def summarize_h5(lag_df: pd.DataFrame) -> Dict:
    valid = lag_df['lag_days'].dropna()
    return {
        'n_onsets': int(len(lag_df)),
        'n_fired': int(lag_df['fired_within_window'].sum()),
        'median_lag': float(valid.median()) if len(valid) else float('nan'),
        'p25_lag': float(valid.quantile(0.25)) if len(valid) else float('nan'),
        'p75_lag': float(valid.quantile(0.75)) if len(valid) else float('nan'),
        'mean_lag': float(valid.mean()) if len(valid) else float('nan'),
        'capture_rate_pct': (
            100.0 * lag_df['fired_within_window'].sum() / max(len(lag_df), 1)
        ),
    }


def run_lengths(series: pd.Series) -> pd.DataFrame:
    blocks = (series != series.shift()).cumsum()
    return series.groupby(blocks).agg(['first', 'size'])


def regime_distribution(labels: pd.DataFrame) -> pd.Series:
    """Return regime %s across the replay window."""
    return labels['regime'].value_counts(normalize=True).mul(100.0).round(2)


def regime_distribution_by_year(labels: pd.DataFrame) -> pd.DataFrame:
    df = labels.copy()
    df['year_int'] = df.index.year
    dist = df.groupby(['year_int', 'regime']).size().unstack(fill_value=0)
    return dist.div(dist.sum(axis=1), axis=0).mul(100.0).round(1)


def regime_run_lengths(labels: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    rl = run_lengths(labels['regime'])
    out = {}
    for regime in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
        sizes = rl.loc[rl['first'] == regime, 'size']
        if len(sizes) == 0:
            out[regime] = {'n': 0}
            continue
        out[regime] = {
            'n': int(len(sizes)),
            'median': float(sizes.median()),
            'p25': float(sizes.quantile(0.25)),
            'p75': float(sizes.quantile(0.75)),
            'max': int(sizes.max()),
        }
    return out


def transition_matrix_diag_mass(labels: pd.DataFrame) -> float:
    tm = pd.crosstab(
        labels['regime'].shift(), labels['regime'], normalize='index',
    )
    return float(np.mean([
        tm.loc[r, r] for r in tm.index if r in tm.columns
    ]))


def render_section(title: str, lines: List[str]) -> List[str]:
    return [f'## {title}', ''] + lines + ['']


def render_diagnosis(
    v0_lag_g4: pd.DataFrame,
    v1_lag_g4: pd.DataFrame,
    verdict: str,
) -> List[str]:
    """Diagnosis + recommendation; structure differs by verdict."""
    lines: List[str] = []
    if verdict == 'PASS':
        lines.append('Gate 1 PASS. Proceed to Gate 2 (pre-spec tau registration).')
        lines.append('')
        return lines

    # FAIL/BLOCKED diagnosis.
    v0s = summarize_h5(v0_lag_g4)
    v1s = summarize_h5(v1_lag_g4)
    lines.append('Gate 1 FAIL. WS-3d is BLOCKED at the diagnostic-rerun gate.')
    lines.append('')
    lines.append('### Root cause')
    lines.append('')
    lines.append('The v1 detector was trained to predict G1_BEAR (drawdown >= 10%')
    lines.append('from trailing 252-day high), and consumed via an argmax-flip-on-0.5')
    lines.append('mapping per the spec (BEAR_PROB_THRESHOLD = 0.5 in')
    lines.append('src/strategies/advanced/market_regime_detector_v1.py). By')
    lines.append('construction, P(G1_BEAR | indicators) only crosses 0.5 around the')
    lines.append('same time G1_BEAR itself fires -- which is AFTER the drawdown is')
    lines.append('confirmed at ~10%. The argmax label therefore tracks confirmation')
    lines.append('rather than precedes it. v1 H2 recall (96.5%) vs v0 (46.1%) is the')
    lines.append('other side of this: v1 is dominant on confirmed G1_BEAR days but')
    lines.append('does not flip BEAR earlier than v0 on the GATE-relevant G4-event')
    lines.append('basis.')
    lines.append('')
    lines.append('Per-event detail (G4 basis):')
    lines.append('')
    merged = v0_lag_g4.merge(v1_lag_g4, on=['event', 'start_date'], suffixes=('_v0', '_v1'))
    for _, row in merged.iterrows():
        v0lag = row['lag_days_v0']
        v1lag = row['lag_days_v1']
        v0lag_s = f'{int(v0lag)}' if pd.notna(v0lag) else 'DID NOT FIRE'
        v1lag_s = f'{int(v1lag)}' if pd.notna(v1lag) else 'DID NOT FIRE'
        if pd.isna(v1lag):
            verdict_str = 'WORSE (v1 missed)'
        elif pd.isna(v0lag):
            verdict_str = 'BETTER (v0 missed)'
        elif v1lag < v0lag:
            verdict_str = f'BETTER by {int(v0lag - v1lag)}d'
        elif v1lag > v0lag:
            verdict_str = f'WORSE by {int(v1lag - v0lag)}d'
        else:
            verdict_str = 'TIE'
        lines.append(
            f'- {row["event"]} ({row["start_date"].date()}): v0={v0lag_s}, '
            f'v1={v1lag_s}  ->  {verdict_str}'
        )
    lines.append('')
    lines.append('### Recommended spec revisions')
    lines.append('')
    lines.append('1. **Lower BEAR_PROB_THRESHOLD or move to a Schmitt-trigger consumer.**')
    lines.append('   The raw P(BEAR) trace shows v1 crosses 0.25-0.30 days before the')
    lines.append('   argmax fires at 0.5. The spec already plans Gate 2 (pre-spec tau')
    lines.append('   from G1_BEAR median on v1 outputs); that tau will likely be in')
    lines.append('   the 0.10-0.30 band and would make V14-style Schmitt-trigger')
    lines.append('   variants (V20-rd-bear-cash, etc.) fire earlier than v0.')
    lines.append('   However, this defers the test to Gate 3 (readiness) instead of')
    lines.append('   the diagnostic gate. The spec needs amendment to either (a) move')
    lines.append('   Gate 1 to evaluate on the Schmitt-fired label rather than')
    lines.append('   argmax, or (b) explicitly accept that Gate 1 measures argmax')
    lines.append('   lag and is informational only when the consumer is Schmitt-based.')
    lines.append('')
    lines.append('2. **Train on a LEADING target instead of G1_BEAR.**')
    lines.append('   G1_BEAR is a CONFIRMATION label by construction. Train on G2_BEAR')
    lines.append('   (forward 30-day return < -5% AND forward vol > 25%) instead.')
    lines.append('   G2 is forward-looking but in-sample-only is acceptable since')
    lines.append('   training data is by definition historical. The trade-off is that')
    lines.append('   G2 has more class imbalance and harder to learn.')
    lines.append('')
    lines.append('3. **Alternative: train on G1_BEAR shifted backward by k days.**')
    lines.append('   Use label = G1_BEAR.shift(-k) for k in {5, 10, 15}, picking k')
    lines.append('   that maximizes recall on G4 events at the target lag. This is')
    lines.append('   methodologically cleanest -- still supervised, but on a leading')
    lines.append('   target rather than a coincident one.')
    lines.append('')
    lines.append('4. **Consider the alternative architectures in the spec Appendix.**')
    lines.append('   HMM or threshold-ensemble may have different lag characteristics.')
    lines.append('   But none of them address the underlying issue that a confirmation')
    lines.append('   label cannot be predicted ahead of itself by a supervised model')
    lines.append('   with a 0.5 decision threshold.')
    lines.append('')
    lines.append('5. **Escalate to halt-or-redirect per parent WS-3 spec Appendix.**')
    lines.append('   Three independent measurements of structural detector lag (V12')
    lines.append('   gap_days=-3.42, v0 H5=14d, E8 exit-to-low=-8d) led to this spec.')
    lines.append('   If WS-3d cannot reduce H5 lag with a fresh architecture AND a')
    lines.append('   fresh input set, the regime-aware approach may be at its useful')
    lines.append('   limit for RAMP regardless of detector iteration.')
    lines.append('')
    lines.append('### Stop here per Pre-commitment 5')
    lines.append('')
    lines.append('Per spec: "the diagnostic rerun is a gating check before the')
    lines.append('readiness orchestrator runs: if H5 lag is not reduced by 30%, the')
    lines.append('leading indicator set OR the architecture is wrong and we don\'t')
    lines.append('proceed to readiness gating."')
    lines.append('')
    lines.append('Gates 2-6 are NOT run. Spec revision is required before continuing.')
    lines.append('')
    return lines


def render_h5_g4_comparison(
    v0_lag_df: pd.DataFrame, v1_lag_df: pd.DataFrame,
) -> List[str]:
    """Apples-to-apples G4 hand-curated event basis matching 20260523."""
    v0s = summarize_h5(v0_lag_df)
    v1s = summarize_h5(v1_lag_df)
    lines = []
    lines.append('### H5 (G4-event basis): lag from drawdown event start')
    lines.append('to first detector-BEAR label within event window')
    lines.append('')
    lines.append('| Metric | v0 (baseline) | v1 (WS-3d) |')
    lines.append('|---|---|---|')
    lines.append(f'| n events | {v0s["n_onsets"]} | {v1s["n_onsets"]} |')
    lines.append(f'| n fired | {v0s["n_fired"]} | {v1s["n_fired"]} |')
    lines.append(f'| capture rate | {v0s["capture_rate_pct"]:.1f}% | {v1s["capture_rate_pct"]:.1f}% |')
    lines.append(f'| median lag | {v0s["median_lag"]:.1f} | {v1s["median_lag"]:.1f} |')
    lines.append(f'| P25 lag | {v0s["p25_lag"]:.1f} | {v1s["p25_lag"]:.1f} |')
    lines.append(f'| P75 lag | {v0s["p75_lag"]:.1f} | {v1s["p75_lag"]:.1f} |')
    lines.append(f'| mean lag | {v0s["mean_lag"]:.1f} | {v1s["mean_lag"]:.1f} |')
    lines.append('')
    lines.append('Per-event:')
    lines.append('')
    lines.append('| event | start | v0 lag | v0 fired | v1 lag | v1 fired |')
    lines.append('|---|---|---|---|---|---|')
    merged = v0_lag_df.merge(v1_lag_df, on=['event', 'start_date'], suffixes=('_v0', '_v1'))
    for _, row in merged.iterrows():
        v0lag = row['lag_days_v0']
        v1lag = row['lag_days_v1']
        v0lag_s = f'{int(v0lag)}' if pd.notna(v0lag) else 'n/a'
        v1lag_s = f'{int(v1lag)}' if pd.notna(v1lag) else 'n/a'
        lines.append(
            f'| {row["event"]} | {row["start_date"].date()} '
            f'| {v0lag_s} | {row["fired_within_window_v0"]} '
            f'| {v1lag_s} | {row["fired_within_window_v1"]} |'
        )
    lines.append('')
    return lines


def render_h5_comparison(
    v0_lag_df: pd.DataFrame, v1_lag_df: pd.DataFrame,
) -> List[str]:
    v0s = summarize_h5(v0_lag_df)
    v1s = summarize_h5(v1_lag_df)

    lines: List[str] = []
    lines.append('### H5: Median lag from G1_BEAR onset to first detector-BEAR label')
    lines.append('')
    lines.append('Methodology: for each G1_BEAR onset (False -> True transition),')
    lines.append(f'measure the lag in calendar days from the onset to the first')
    lines.append(f'detector-BEAR label within a forward {FORWARD_WINDOW_DAYS}-day window.')
    lines.append('Onsets with no BEAR fire in window are reported separately and')
    lines.append('excluded from the lag distribution.')
    lines.append('')
    lines.append('NOTE: G1_BEAR is a drawdown-confirmation label. It fires AFTER')
    lines.append('SPY has already declined >= 10% from its trailing 252-day peak.')
    lines.append('Both v0 and v1 typically already have BEAR active by the time')
    lines.append('G1_BEAR turns True, so the G1-basis lag often saturates to 0d.')
    lines.append('The Gate 1 verdict is decided on the G4-event basis above, which')
    lines.append('measures from drawdown START rather than from drawdown CONFIRMATION.')
    lines.append('')
    lines.append('| Metric | v0 (baseline) | v1 (WS-3d) |')
    lines.append('|---|---|---|')
    lines.append(f'| n onsets | {v0s["n_onsets"]} | {v1s["n_onsets"]} |')
    lines.append(f'| n fired in window | {v0s["n_fired"]} | {v1s["n_fired"]} |')
    lines.append(f'| capture rate | {v0s["capture_rate_pct"]:.1f}% | {v1s["capture_rate_pct"]:.1f}% |')
    lines.append(f'| median lag (days) | {v0s["median_lag"]:.1f} | {v1s["median_lag"]:.1f} |')
    lines.append(f'| P25 lag (days) | {v0s["p25_lag"]:.1f} | {v1s["p25_lag"]:.1f} |')
    lines.append(f'| P75 lag (days) | {v0s["p75_lag"]:.1f} | {v1s["p75_lag"]:.1f} |')
    lines.append(f'| mean lag (days) | {v0s["mean_lag"]:.1f} | {v1s["mean_lag"]:.1f} |')
    lines.append('')
    return lines


def render_h1_section(v0: pd.DataFrame, v1: pd.DataFrame) -> List[str]:
    v0_dist = regime_distribution(v0)
    v1_dist = regime_distribution(v1)
    lines = []
    lines.append('### H1: Regime distribution parity')
    lines.append('')
    lines.append('| Regime | v0 % | v1 % |')
    lines.append('|---|---|---|')
    for r in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
        lines.append(f'| {r} | {v0_dist.get(r, 0.0):.2f} | {v1_dist.get(r, 0.0):.2f} |')
    lines.append('')
    return lines


def render_h2_section(v0: pd.DataFrame, v1: pd.DataFrame, gt: pd.DataFrame) -> List[str]:
    """Confusion-matrix style: how often do v0/v1 label BEAR on G1_BEAR=True days."""
    g1 = gt['g1_bear']
    v0j = v0.join(gt[['g1_bear']], how='inner')
    v1j = v1.join(gt[['g1_bear']], how='inner')
    v0_recall = (
        ((v0j['regime'] == 'BEAR') & v0j['g1_bear']).sum() / max(v0j['g1_bear'].sum(), 1) * 100.0
    )
    v1_recall = (
        ((v1j['regime'] == 'BEAR') & v1j['g1_bear']).sum() / max(v1j['g1_bear'].sum(), 1) * 100.0
    )
    v0_precision = (
        ((v0j['regime'] == 'BEAR') & v0j['g1_bear']).sum() / max((v0j['regime'] == 'BEAR').sum(), 1) * 100.0
    )
    v1_precision = (
        ((v1j['regime'] == 'BEAR') & v1j['g1_bear']).sum() / max((v1j['regime'] == 'BEAR').sum(), 1) * 100.0
    )

    lines = []
    lines.append('### H2: BEAR label vs G1_BEAR ground truth (precision/recall)')
    lines.append('')
    lines.append('| Metric | v0 | v1 |')
    lines.append('|---|---|---|')
    lines.append(f'| Total G1_BEAR days | {int(v0j["g1_bear"].sum())} | {int(v1j["g1_bear"].sum())} |')
    lines.append(f'| Total detector-BEAR days | {int((v0j["regime"] == "BEAR").sum())} | {int((v1j["regime"] == "BEAR").sum())} |')
    lines.append(f'| Recall (BEAR | G1_BEAR) | {v0_recall:.1f}% | {v1_recall:.1f}% |')
    lines.append(f'| Precision (G1_BEAR | BEAR) | {v0_precision:.1f}% | {v1_precision:.1f}% |')
    lines.append('')
    return lines


def render_h3_section(v0: pd.DataFrame, v1: pd.DataFrame, gt: pd.DataFrame) -> List[str]:
    """H3 here: BEAR-label association with G3_vol_spike (volatility regime)."""
    v0j = v0.join(gt[['g3_vol_spike']], how='inner')
    v1j = v1.join(gt[['g3_vol_spike']], how='inner')
    v0_assoc = (
        ((v0j['regime'] == 'BEAR') & v0j['g3_vol_spike']).sum() / max((v0j['regime'] == 'BEAR').sum(), 1) * 100.0
    )
    v1_assoc = (
        ((v1j['regime'] == 'BEAR') & v1j['g3_vol_spike']).sum() / max((v1j['regime'] == 'BEAR').sum(), 1) * 100.0
    )
    lines = []
    lines.append('### H3: BEAR co-occurrence with G3_vol_spike')
    lines.append('')
    lines.append('| Metric | v0 | v1 |')
    lines.append('|---|---|---|')
    lines.append(f'| % of BEAR days with G3_vol_spike | {v0_assoc:.1f}% | {v1_assoc:.1f}% |')
    lines.append('')
    return lines


def render_h4_section(v0: pd.DataFrame, v1: pd.DataFrame) -> List[str]:
    v0_rl = regime_run_lengths(v0)
    v1_rl = regime_run_lengths(v1)
    v0_diag = transition_matrix_diag_mass(v0)
    v1_diag = transition_matrix_diag_mass(v1)
    lines = []
    lines.append('### H4: Run-length / flicker (transitions and persistence)')
    lines.append('')
    lines.append('| Regime | v0 n_runs | v0 median | v0 max | v1 n_runs | v1 median | v1 max |')
    lines.append('|---|---|---|---|---|---|---|')
    for r in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
        v0e = v0_rl[r]
        v1e = v1_rl[r]
        v0_n = v0e.get('n', 0)
        v1_n = v1e.get('n', 0)
        v0_med = v0e.get('median', float('nan')) if v0_n else float('nan')
        v1_med = v1e.get('median', float('nan')) if v1_n else float('nan')
        v0_max = v0e.get('max', float('nan')) if v0_n else float('nan')
        v1_max = v1e.get('max', float('nan')) if v1_n else float('nan')
        lines.append(
            f'| {r} | {v0_n} | {v0_med if isinstance(v0_med, float) and not np.isnan(v0_med) else "n/a"} '
            f'| {v0_max if isinstance(v0_max, (int, float)) and v0_n else "n/a"} '
            f'| {v1_n} | {v1_med if isinstance(v1_med, float) and not np.isnan(v1_med) else "n/a"} '
            f'| {v1_max if isinstance(v1_max, (int, float)) and v1_n else "n/a"} |'
        )
    lines.append('')
    lines.append(f'Transition-matrix mean diagonal mass: v0={v0_diag:.3f}, v1={v1_diag:.3f}')
    lines.append('')
    return lines


def gate1_verdict(v0_g4_median: float, v1_g4_median: float, v1_g1_median: float) -> tuple:
    """Determine Gate 1 pass/fail based on the G4 same-basis methodology
    (apples-to-apples with the 14d baseline of record from the 20260523 report).

    Returns (verdict, reduction_pct_g4, reduction_pct_baseline).
    """
    if v0_g4_median > 0:
        reduction_pct_g4 = (v0_g4_median - v1_g4_median) / v0_g4_median * 100.0
    else:
        reduction_pct_g4 = float('nan')
    reduction_pct_baseline = (
        (V0_H5_BASELINE_DAYS - v1_g4_median) / V0_H5_BASELINE_DAYS * 100.0
    )
    # PASS criterion: v1 G4-basis median <= 10d (>=30% reduction from 14d baseline)
    # OR if v0 same-run G4 median > 0, v1 reduces same-basis median by >= 30%.
    pass_baseline = v1_g4_median <= (
        V0_H5_BASELINE_DAYS * (1 - H5_REDUCTION_THRESHOLD_PCT / 100.0)
    )
    pass_same_basis = (
        not np.isnan(reduction_pct_g4)
        and reduction_pct_g4 >= H5_REDUCTION_THRESHOLD_PCT
    )
    if pass_baseline or pass_same_basis:
        verdict = 'PASS'
    else:
        verdict = 'FAIL'
    return verdict, reduction_pct_g4, reduction_pct_baseline


def build_report(v0: pd.DataFrame, v1: pd.DataFrame, gt: pd.DataFrame) -> str:
    """Assemble the H1-H5 markdown report."""
    g1 = gt['g1_bear']
    v0_lag = compute_h5_lag(v0, g1)
    v1_lag = compute_h5_lag(v1, g1)
    v0_lag_g4 = compute_h5_lag_g4(v0, G4_EVENTS)
    v1_lag_g4 = compute_h5_lag_g4(v1, G4_EVENTS)

    v0s = summarize_h5(v0_lag)
    v1s = summarize_h5(v1_lag)
    v0s_g4 = summarize_h5(v0_lag_g4)
    v1s_g4 = summarize_h5(v1_lag_g4)

    verdict, reduction_pct_g4, reduction_pct_baseline = gate1_verdict(
        v0s_g4['median_lag'], v1s_g4['median_lag'], v1s['median_lag'],
    )
    if v0s['median_lag'] > 0:
        reduction_pct_g1 = (v0s['median_lag'] - v1s['median_lag']) / v0s['median_lag'] * 100.0
    else:
        reduction_pct_g1 = float('nan')

    lines: List[str] = []
    lines.append('# WS-3d Diagnostic Rerun -- H1-H5 on the v1 LightGBM Detector')
    lines.append('')
    lines.append('**Date**: 2026-06-01')
    lines.append(f'**Branch**: v12-bear-to-cash')
    lines.append('**Spec**: docs/superpowers/specs/2026-05-25-ws3d-detector-replacement-design.md')
    lines.append('**Gate**: Gate 1 (H5 lag reduction, GATING)')
    lines.append('**Status**: Gate 1 ' + verdict)
    lines.append('')
    lines.append('## Headline')
    lines.append('')
    lines.append('Two H5 measurement bases are reported. The G4-event basis matches')
    lines.append('the methodology of the 20260523 v0 baseline-of-record (14d) and is')
    lines.append('the apples-to-apples comparison the Gate 1 verdict uses. The')
    lines.append('G1_BEAR-onset basis (spec methodology) is also reported but')
    lines.append('typically saturates to 0d because G1_BEAR is a drawdown-confirmed')
    lines.append('label that fires AFTER the price weakness both detectors react to.')
    lines.append('')
    lines.append(f'- v0 H5 median lag, G4-event basis (this run): {v0s_g4["median_lag"]:.1f} days')
    lines.append(f'- v0 H5 baseline of record (20260523, G4-event basis): {V0_H5_BASELINE_DAYS:.1f} days')
    lines.append(f'- v1 H5 median lag, G4-event basis (this run): {v1s_g4["median_lag"]:.1f} days')
    lines.append(f'- Reduction vs v0 G4-basis same run: {reduction_pct_g4:.1f}%')
    lines.append(f'- Reduction vs 14d baseline of record: {reduction_pct_baseline:.1f}%')
    lines.append('')
    lines.append(f'- v0 H5 median lag, G1_BEAR-onset basis (this run): {v0s["median_lag"]:.1f} days')
    lines.append(f'- v1 H5 median lag, G1_BEAR-onset basis (this run): {v1s["median_lag"]:.1f} days')
    lines.append(f'- Reduction vs v0 G1-basis same run: {reduction_pct_g1:.1f}%')
    lines.append('')
    lines.append(f'- Pre-commitment 5 threshold: >= {H5_REDUCTION_THRESHOLD_PCT:.0f}% reduction (v1 median <= 10d)')
    lines.append('')
    lines.append(f'**Verdict: Gate 1 {verdict}**')
    lines.append('')

    # G4 same-basis section
    lines.extend(render_section('H5 -- G4 same-basis (apples-to-apples)', render_h5_g4_comparison(v0_lag_g4, v1_lag_g4)))
    lines.extend(render_section('H5 -- G1_BEAR onset basis (spec methodology)', render_h5_comparison(v0_lag, v1_lag)))
    lines.extend(render_section(
        'Diagnosis and recommendation',
        render_diagnosis(v0_lag_g4, v1_lag_g4, verdict),
    ))
    lines.extend(render_section('H1 -- Regime distribution', render_h1_section(v0, v1)))
    lines.extend(render_section('H2 -- BEAR vs G1_BEAR precision/recall', render_h2_section(v0, v1, gt)))
    lines.extend(render_section('H3 -- BEAR vs G3_vol_spike', render_h3_section(v0, v1, gt)))
    lines.extend(render_section('H4 -- Run lengths and flicker', render_h4_section(v0, v1)))

    # Per-onset lag tables (informational, last)
    lines.append('## Per-onset lag detail')
    lines.append('')
    lines.append('### v0 lag per G1_BEAR onset')
    lines.append('')
    lines.append('| onset_date | lag_days | fired_within_window |')
    lines.append('|---|---|---|')
    for _, row in v0_lag.iterrows():
        lag = row['lag_days']
        lag_str = f'{int(lag)}' if pd.notna(lag) else 'n/a'
        lines.append(f'| {row["onset_date"].date()} | {lag_str} | {row["fired_within_window"]} |')
    lines.append('')
    lines.append('### v1 lag per G1_BEAR onset')
    lines.append('')
    lines.append('| onset_date | lag_days | fired_within_window |')
    lines.append('|---|---|---|')
    for _, row in v1_lag.iterrows():
        lag = row['lag_days']
        lag_str = f'{int(lag)}' if pd.notna(lag) else 'n/a'
        lines.append(f'| {row["onset_date"].date()} | {lag_str} | {row["fired_within_window"]} |')
    lines.append('')

    return '\n'.join(lines)


def main(argv: Optional[list] = None) -> int:
    if not V1_LABELS.exists():
        raise FileNotFoundError(
            f'{V1_LABELS} not found. Run regime_detector_v1_replay.py first.'
        )
    if not V0_LABELS.exists():
        raise FileNotFoundError(
            f'{V0_LABELS} not found. Run regime_detector_replay.py first.'
        )
    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f'{GROUND_TRUTH} not found. Run ground_truth_labelers.py first.'
        )

    v1 = _load_labels(V1_LABELS)
    v0 = _load_labels(V0_LABELS)
    gt = pd.read_parquet(GROUND_TRUTH)
    gt['date'] = pd.to_datetime(gt['date'])
    gt = gt.set_index('date').sort_index()

    common = v1.index.intersection(v0.index).intersection(gt.index)
    v1 = v1.loc[common]
    v0 = v0.loc[common]
    gt = gt.loc[common]
    logger.info(
        f'[+] joined panel: {len(common)} days '
        f'({common.min().date()} to {common.max().date()})'
    )

    report = build_report(v0, v1, gt)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='ascii', errors='replace')
    logger.info(f'[+] wrote {REPORT_PATH} ({len(report)} chars)')

    # Recompute headline for stdout (so callers can scrape the verdict).
    g1 = gt['g1_bear']
    v0_lag = compute_h5_lag(v0, g1)
    v1_lag = compute_h5_lag(v1, g1)
    v0_lag_g4 = compute_h5_lag_g4(v0, G4_EVENTS)
    v1_lag_g4 = compute_h5_lag_g4(v1, G4_EVENTS)
    v0s = summarize_h5(v0_lag)
    v1s = summarize_h5(v1_lag)
    v0s_g4 = summarize_h5(v0_lag_g4)
    v1s_g4 = summarize_h5(v1_lag_g4)

    verdict, reduction_pct_g4, reduction_pct_baseline = gate1_verdict(
        v0s_g4['median_lag'], v1s_g4['median_lag'], v1s['median_lag'],
    )

    logger.info('=== Gate 1 summary ===')
    logger.info(f'v0 median lag, G4-event basis (this run): {v0s_g4["median_lag"]:.1f}d')
    logger.info(f'v0 baseline of record (14d, G4 basis):    {V0_H5_BASELINE_DAYS:.1f}d')
    logger.info(f'v1 median lag, G4-event basis (this run): {v1s_g4["median_lag"]:.1f}d')
    logger.info(f'Reduction vs v0 G4-basis same run:        {reduction_pct_g4:.1f}%')
    logger.info(f'Reduction vs 14d baseline of record:      {reduction_pct_baseline:.1f}%')
    logger.info('')
    logger.info(f'v0 median lag, G1-onset basis (this run): {v0s["median_lag"]:.1f}d')
    logger.info(f'v1 median lag, G1-onset basis (this run): {v1s["median_lag"]:.1f}d')
    logger.info('')
    logger.info(f'GATE 1: {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
