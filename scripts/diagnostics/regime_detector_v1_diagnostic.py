"""WS-3d diagnostic rerun: H1-H5 analyses on the v1 detector vs the v0 baseline.

Reads:
- diagnostics/regime/v1/labels.parquet (v1 replay output)
- diagnostics/regime/v0/labels.parquet (v0 replay output, locked baseline)
- diagnostics/regime/v0_scores/labels.parquet (v0 replay with continuous score_BEAR)
- diagnostics/regime/ground_truth.parquet (G1_BEAR, G2, G3, G4)

H5 measurement (Pre-commitment 5, GATING):
- For every G4 hand-curated drawdown-event start, measure the lag in calendar
  days from the start date to the first detector-BEAR signal within the event
  window. Two evaluation methodologies are reported:
    - argmax-at-0.5 (legacy v0-style, informational post-Amendment 6)
    - Schmitt-tau-0.30 first-crossing (binding for Gate 1 verdict, Amendment 6)
  v1 lag must be >= 30% lower than v0 lag at the binding metric.

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
V0_SCORES_LABELS = Path('diagnostics/regime/v0_scores/labels.parquet')
GROUND_TRUTH = Path('diagnostics/regime/ground_truth.parquet')
G4_EVENTS = Path('config/diagnostics/regime_events_2017_2026.csv')
REPORT_PATH = Path('docs/reports/ramp/20260601_ws3d_regime_diagnostic_rerun.md')

FORWARD_WINDOW_DAYS = 60  # forward lag-search window per spec
V0_H5_BASELINE_DAYS = 14.0  # from 20260523_regime_detector_diagnostic.md (argmax-at-0.5 baseline)
H5_REDUCTION_THRESHOLD_PCT = 30.0  # >= 30% reduction required to pass Gate 1

# Amendment 6 (2026-05-24): Schmitt-trigger evaluation tau. Pre-registered from
# E3's soft-score lead-time finding (tau=0.3 was best in v0's soft-score sweep).
TAU_EVAL = 0.30
V0_SCORE_BEAR_COL = 'score_BEAR'  # v0_scores schema (rule-based fraction in [0, 1])
V1_BEAR_PROBA_COL = 'bear_proba'  # v1 schema (LightGBM predict_proba in [0, 1])


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


def compute_h5_schmitt_lag_g4(
    labels: pd.DataFrame,
    score_col: str,
    events_csv: Path,
    tau_eval: float = TAU_EVAL,
    event_type: str = 'drawdown',
) -> pd.DataFrame:
    """H5 metric (Amendment 6, 2026-05-24): median lag from G4 hand-curated
    drawdown-event start to first day where score_col >= tau_eval.

    Uses the same event-window cap [start_date, end_date] as the existing
    argmax-at-0.5 G4 metric so the two are directly comparable.

    Parameters
    ----------
    labels : pd.DataFrame
        Indexed by date. Must contain score_col with a probability in [0, 1].
    score_col : str
        Column name of the continuous BEAR score.
        - v1: 'bear_proba' (LightGBM predict_proba[..., 1])
        - v0_scores: 'score_BEAR' (rule-based fraction-of-criteria-met)
    events_csv : Path
        G4 hand-curated events CSV.
    tau_eval : float
        Schmitt-trigger threshold for first-crossing detection.
    event_type : str
        Filter on event_type column (default 'drawdown').

    Returns
    -------
    pd.DataFrame with [event, start_date, lag_days, fired_within_window].
    """
    if score_col not in labels.columns:
        raise KeyError(
            f"H5 Schmitt eval requires column {score_col!r} in labels; "
            f"available: {list(labels.columns)}"
        )
    events = pd.read_csv(events_csv, parse_dates=['start_date', 'end_date'])
    events = events[events['event_type'] == event_type]
    rows = []
    for _, ev in events.iterrows():
        window = labels.loc[ev['start_date']:ev['end_date']]
        scores = window[score_col]
        crossings = window.index[scores >= tau_eval]
        if len(crossings) == 0:
            rows.append({
                'event': ev['event_name'],
                'start_date': ev['start_date'],
                'lag_days': float('nan'),
                'fired_within_window': False,
            })
        else:
            lag = (crossings[0] - ev['start_date']).days
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
    """Diagnosis + recommendation; structure differs by verdict.

    Note: post-Amendment 6, `verdict` is the Schmitt-tau-0.30 binding verdict.
    The diagnosis text below was originally written for the argmax-at-0.5
    failure mode; it remains relevant because the argmax-at-0.5 BLOCKED finding
    is still on the record and the spec's fallback options (b)/(c)/(d) listed
    in Amendment 6's last paragraph mirror the recommendations here.
    """
    lines: List[str] = []
    if verdict == 'PASS':
        lines.append('Gate 1 PASS (Amendment 6 binding metric). Proceed to Gate 2 (pre-spec tau registration).')
        lines.append('')
        return lines

    # FAIL/BLOCKED diagnosis.
    v0s = summarize_h5(v0_lag_g4)
    v1s = summarize_h5(v1_lag_g4)
    lines.append('Gate 1 FAIL. WS-3d is BLOCKED at the diagnostic-rerun gate (Amendment 6 binding metric).')
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
    lines.append('Amendment 6 (Schmitt-tau-0.30 first-crossing) tested the alternative')
    lines.append('that v1\'s probabilistic score crosses the consumer-layer threshold')
    lines.append('earlier than the argmax flips at 0.5. It does -- v1\'s P(BEAR)')
    lines.append('reaches 0.30 days before reaching 0.5 -- but v0\'s rule-based')
    lines.append('`score_BEAR` is a fraction-of-criteria-met that flips immediately on')
    lines.append('the first price-decline criterion (above_20, above_50, above_200,')
    lines.append('vix_percentile). At tau=0.30, v0\'s score crosses on day 1-2 of most')
    lines.append('drawdowns, while v1\'s P(BEAR) still takes weeks. The Schmitt-trigger')
    lines.append('reformulation does not rescue Gate 1: v1 is slower than v0 at tau=0.30')
    lines.append('on the binding metric.')
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
    """Legacy (argmax-at-0.5) Gate 1 verdict, retained for transparency.

    After Amendment 6 (2026-05-24) this is NO LONGER the binding metric.
    Use gate1_verdict_schmitt() for the post-amendment binding decision.

    Returns (verdict, reduction_pct_g4, reduction_pct_baseline).
    """
    if v0_g4_median > 0:
        reduction_pct_g4 = (v0_g4_median - v1_g4_median) / v0_g4_median * 100.0
    else:
        reduction_pct_g4 = float('nan')
    reduction_pct_baseline = (
        (V0_H5_BASELINE_DAYS - v1_g4_median) / V0_H5_BASELINE_DAYS * 100.0
    )
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


def gate1_verdict_schmitt(v0_schmitt_median: float, v1_schmitt_median: float) -> tuple:
    """Amendment 6 binding Gate 1 verdict: Schmitt-tau-0.30 first-crossing on G4.

    PASS criterion (apples-to-apples, both detectors evaluated at tau_eval=0.30):
      - v1 median lag <= 10 calendar days (>=30% reduction from a 14d notional ceiling)
      - AND v1 reduces v0's same-basis median by >= 30%

    The v0 14d argmax-at-0.5 baseline is NOT used here -- Amendment 6 requires the
    same metric on both detectors. v0's Schmitt-tau-0.30 median is the honest
    baseline.

    Returns (verdict, reduction_pct_v0_same_basis, v1_meets_absolute_floor).
    """
    if not np.isnan(v0_schmitt_median) and v0_schmitt_median > 0:
        reduction_pct = (v0_schmitt_median - v1_schmitt_median) / v0_schmitt_median * 100.0
    else:
        reduction_pct = float('nan')
    # Absolute floor: v1 must be at or below 10d (30% off 14d argmax-at-0.5 ceiling).
    v1_meets_floor = (
        not np.isnan(v1_schmitt_median)
        and v1_schmitt_median <= V0_H5_BASELINE_DAYS * (1 - H5_REDUCTION_THRESHOLD_PCT / 100.0)
    )
    v1_reduces_v0 = (
        not np.isnan(reduction_pct)
        and reduction_pct >= H5_REDUCTION_THRESHOLD_PCT
    )
    if v1_meets_floor and v1_reduces_v0:
        verdict = 'PASS'
    else:
        verdict = 'FAIL'
    return verdict, reduction_pct, v1_meets_floor


def build_report(
    v0: pd.DataFrame,
    v1: pd.DataFrame,
    gt: pd.DataFrame,
    v0_scores: pd.DataFrame,
) -> str:
    """Assemble the H1-H5 markdown report."""
    g1 = gt['g1_bear']
    v0_lag = compute_h5_lag(v0, g1)
    v1_lag = compute_h5_lag(v1, g1)
    v0_lag_g4 = compute_h5_lag_g4(v0, G4_EVENTS)
    v1_lag_g4 = compute_h5_lag_g4(v1, G4_EVENTS)

    # Amendment 6: Schmitt-tau-0.30 first-crossing on G4 drawdown events.
    v0_lag_schmitt = compute_h5_schmitt_lag_g4(
        v0_scores, V0_SCORE_BEAR_COL, G4_EVENTS, TAU_EVAL,
    )
    v1_lag_schmitt = compute_h5_schmitt_lag_g4(
        v1, V1_BEAR_PROBA_COL, G4_EVENTS, TAU_EVAL,
    )

    v0s = summarize_h5(v0_lag)
    v1s = summarize_h5(v1_lag)
    v0s_g4 = summarize_h5(v0_lag_g4)
    v1s_g4 = summarize_h5(v1_lag_g4)
    v0s_sch = summarize_h5(v0_lag_schmitt)
    v1s_sch = summarize_h5(v1_lag_schmitt)

    # Legacy argmax-at-0.5 verdict (informational post-Amendment 6).
    verdict_argmax, reduction_pct_g4_argmax, reduction_pct_baseline_argmax = gate1_verdict(
        v0s_g4['median_lag'], v1s_g4['median_lag'], v1s['median_lag'],
    )
    # Amendment 6 binding verdict.
    verdict, reduction_pct_schmitt, v1_meets_floor = gate1_verdict_schmitt(
        v0s_sch['median_lag'], v1s_sch['median_lag'],
    )
    if v0s['median_lag'] > 0:
        reduction_pct_g1 = (v0s['median_lag'] - v1s['median_lag']) / v0s['median_lag'] * 100.0
    else:
        reduction_pct_g1 = float('nan')

    lines: List[str] = []
    lines.append('# WS-3d Diagnostic Rerun -- H1-H5 on the v1 LightGBM Detector')
    lines.append('')
    lines.append('**Date**: 2026-06-01 (Amendment 6 applied 2026-05-24)')
    lines.append(f'**Branch**: v12-bear-to-cash')
    lines.append('**Spec**: docs/superpowers/specs/2026-05-25-ws3d-detector-replacement-design.md')
    lines.append('**Gate**: Gate 1 (H5 lag reduction, GATING)')
    lines.append('**Status**: Gate 1 ' + verdict + ' (binding metric: Schmitt-tau-0.30)')
    lines.append('')

    # ---- Amendment 6 section (placed at top per task spec) ----
    lines.append('## Amendment 6 (2026-05-24) -- Gate 1 H5 metric revision')
    lines.append('')
    lines.append('The original Gate 1 ran against argmax-at-0.5 evaluation (v0-historical')
    lines.append('convention). Result: v1 H5 lag = 21 days vs v0 14 days (BLOCKED).')
    lines.append('')
    lines.append('Amendment 6 changes Gate 1\'s H5 metric to Schmitt-trigger first-crossing')
    lines.append('at tau_eval = 0.30, matching the consumer-layer Schmitt-trigger pattern')
    lines.append('V20-rd-bear-cash will use. tau_eval = 0.30 is pre-registered from E3\'s')
    lines.append('soft-score lead-time finding (the same tau that maximized lead-time')
    lines.append('coverage in v0\'s soft-score replay).')
    lines.append('')
    lines.append('Per-detector continuous score columns:')
    lines.append('- v0: `score_BEAR` from `diagnostics/regime/v0_scores/labels.parquet`')
    lines.append('  (rule-based fraction-of-criteria-met, range [0, 1])')
    lines.append('- v1: `bear_proba` from `diagnostics/regime/v1/labels.parquet`')
    lines.append('  (LightGBM `predict_proba(X)[:, 1]`, range [0, 1])')
    lines.append('')
    lines.append('### Updated H5 results')
    lines.append('')
    lines.append('| Metric | v0 | v1 (WS-3d) | Delta | Pass criterion |')
    lines.append('|---|---|---|---|---|')
    lines.append(
        f'| Median H5 lag at argmax-at-0.5 (legacy) | {v0s_g4["median_lag"]:.1f}d '
        f'| {v1s_g4["median_lag"]:.1f}d | {reduction_pct_g4_argmax:.1f}% '
        f'| (informational only post-Amendment 6) |'
    )
    lines.append(
        f'| **Median H5 lag at Schmitt-tau-0.30 (binding)** '
        f'| {v0s_sch["median_lag"]:.1f}d | {v1s_sch["median_lag"]:.1f}d '
        f'| {reduction_pct_schmitt:.1f}% '
        f'| **>= 30% reduction (lag <= 10d) for Gate 1 PASS** |'
    )
    lines.append('')

    # Per-event Schmitt-tau-0.30 table
    lines.append('Per-event G4 detail at Schmitt-tau-0.30:')
    lines.append('')
    lines.append('| event | start | v0 lag (tau=0.30) | v0 fired | v1 lag (tau=0.30) | v1 fired |')
    lines.append('|---|---|---|---|---|---|')
    merged_sch = v0_lag_schmitt.merge(
        v1_lag_schmitt, on=['event', 'start_date'], suffixes=('_v0', '_v1'),
    )
    for _, row in merged_sch.iterrows():
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

    # Combined per-event comparison (argmax AND Schmitt)
    lines.append('Per-event G4 detail -- argmax-at-0.5 AND Schmitt-tau-0.30:')
    lines.append('')
    lines.append('| event | start | v0 argmax | v1 argmax | v0 tau=0.30 | v1 tau=0.30 |')
    lines.append('|---|---|---|---|---|---|')
    merged_arg = v0_lag_g4.merge(
        v1_lag_g4, on=['event', 'start_date'], suffixes=('_v0', '_v1'),
    )
    combo = merged_arg.merge(
        merged_sch, on=['event', 'start_date'], suffixes=('_arg', '_sch'),
    )
    for _, row in combo.iterrows():
        def s(col):
            v = row[col]
            return f'{int(v)}' if pd.notna(v) else 'n/a'
        lines.append(
            f'| {row["event"]} | {row["start_date"].date()} '
            f'| {s("lag_days_v0_arg")} | {s("lag_days_v1_arg")} '
            f'| {s("lag_days_v0_sch")} | {s("lag_days_v1_sch")} |'
        )
    lines.append('')

    # Verdict block
    lines.append('### Gate 1 verdict (Amendment 6)')
    lines.append('')
    if verdict == 'PASS':
        lines.append(
            f'**Gate 1 PASSED.** v1 median Schmitt-tau-0.30 lag = '
            f'{v1s_sch["median_lag"]:.1f}d, '
            f'a {reduction_pct_schmitt:.1f}% reduction from v0\'s {v0s_sch["median_lag"]:.1f}d.'
        )
        lines.append('Proceed to Gate 2 (pre-spec tau registration).')
    else:
        lines.append(
            f'**Gate 1 STILL BLOCKED.** v1 median Schmitt-tau-0.30 lag = '
            f'{v1s_sch["median_lag"]:.1f}d, '
            f'v0 = {v0s_sch["median_lag"]:.1f}d, '
            f'reduction = {reduction_pct_schmitt:.1f}% '
            f'(threshold: >= {H5_REDUCTION_THRESHOLD_PCT:.0f}% AND v1 <= 10d).'
        )
        lines.append('')
        lines.append('Under Amendment 6\'s binding metric, v1\'s LightGBM `bear_proba`')
        lines.append('does not cross tau=0.30 earlier than v0\'s rule-based `score_BEAR`')
        lines.append('on the G4 drawdown events. The leading-indicator hypothesis is in')
        lines.append('doubt. Per Amendment 6\'s last paragraph, the spec falls back to')
        lines.append('one of:')
        lines.append('')
        lines.append('  (b) retrain on a leading target (G2_BEAR forward-window, or')
        lines.append('      G1_BEAR.shift(-k) for k in {5, 10, 15})')
        lines.append('  (c) try the fallback architectures (HMM, threshold ensemble)')
        lines.append('  (d) halt WS-3d -- accept that regime-aware approach may be at')
        lines.append('      its useful limit for RAMP')
        lines.append('')
        lines.append('The decision is the user\'s, not automatic.')
    lines.append('')

    lines.append('## Headline')
    lines.append('')
    lines.append('Three H5 measurement bases are reported. The Schmitt-tau-0.30 G4-event')
    lines.append('basis is the binding metric for Gate 1 (Amendment 6). The argmax-at-0.5')
    lines.append('G4-event basis matches the legacy 20260523 v0 baseline-of-record (14d)')
    lines.append('and is informational only post-Amendment 6. The G1_BEAR-onset basis')
    lines.append('typically saturates to 0d because G1_BEAR is a drawdown-confirmed label')
    lines.append('that fires AFTER the price weakness both detectors react to.')
    lines.append('')
    lines.append('Amendment 6 binding metric (Schmitt-tau-0.30 first-crossing on G4):')
    lines.append(f'- v0 H5 median lag at tau=0.30: {v0s_sch["median_lag"]:.1f} days')
    lines.append(f'- v1 H5 median lag at tau=0.30: {v1s_sch["median_lag"]:.1f} days')
    lines.append(f'- Reduction (v0 same-basis): {reduction_pct_schmitt:.1f}%')
    lines.append(f'- v1 meets <= 10d absolute floor: {v1_meets_floor}')
    lines.append('')
    lines.append('Legacy argmax-at-0.5 metric (informational only post-Amendment 6):')
    lines.append(f'- v0 H5 median lag, G4-event basis (this run): {v0s_g4["median_lag"]:.1f} days')
    lines.append(f'- v0 H5 baseline of record (20260523, G4-event basis): {V0_H5_BASELINE_DAYS:.1f} days')
    lines.append(f'- v1 H5 median lag, G4-event basis (this run): {v1s_g4["median_lag"]:.1f} days')
    lines.append(f'- Reduction vs v0 G4-basis same run: {reduction_pct_g4_argmax:.1f}%')
    lines.append(f'- Reduction vs 14d baseline of record: {reduction_pct_baseline_argmax:.1f}%')
    lines.append(f'- Legacy argmax-at-0.5 verdict (informational): {verdict_argmax}')
    lines.append('')
    lines.append('G1_BEAR-onset basis (typically saturates to 0d):')
    lines.append(f'- v0 H5 median lag: {v0s["median_lag"]:.1f} days')
    lines.append(f'- v1 H5 median lag: {v1s["median_lag"]:.1f} days')
    lines.append(f'- Reduction vs v0 G1-basis same run: {reduction_pct_g1:.1f}%')
    lines.append('')
    lines.append(f'- Pre-commitment 5 threshold: >= {H5_REDUCTION_THRESHOLD_PCT:.0f}% reduction (v1 median <= 10d)')
    lines.append('')
    lines.append(f'**Verdict (Amendment 6 binding): Gate 1 {verdict}**')
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
    if not V0_SCORES_LABELS.exists():
        raise FileNotFoundError(
            f'{V0_SCORES_LABELS} not found. Run regime_score_replay.py first '
            f'(provides v0 continuous score_BEAR for Amendment 6 evaluation).'
        )
    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f'{GROUND_TRUTH} not found. Run ground_truth_labelers.py first.'
        )

    v1 = _load_labels(V1_LABELS)
    v0 = _load_labels(V0_LABELS)
    v0_scores = _load_labels(V0_SCORES_LABELS)
    gt = pd.read_parquet(GROUND_TRUTH)
    gt['date'] = pd.to_datetime(gt['date'])
    gt = gt.set_index('date').sort_index()

    common = (
        v1.index
        .intersection(v0.index)
        .intersection(v0_scores.index)
        .intersection(gt.index)
    )
    v1 = v1.loc[common]
    v0 = v0.loc[common]
    v0_scores = v0_scores.loc[common]
    gt = gt.loc[common]
    logger.info(
        f'[+] joined panel: {len(common)} days '
        f'({common.min().date()} to {common.max().date()})'
    )

    report = build_report(v0, v1, gt, v0_scores)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='ascii', errors='replace')
    logger.info(f'[+] wrote {REPORT_PATH} ({len(report)} chars)')

    # Recompute headline for stdout (so callers can scrape the verdict).
    g1 = gt['g1_bear']
    v0_lag = compute_h5_lag(v0, g1)
    v1_lag = compute_h5_lag(v1, g1)
    v0_lag_g4 = compute_h5_lag_g4(v0, G4_EVENTS)
    v1_lag_g4 = compute_h5_lag_g4(v1, G4_EVENTS)
    v0_lag_sch = compute_h5_schmitt_lag_g4(
        v0_scores, V0_SCORE_BEAR_COL, G4_EVENTS, TAU_EVAL,
    )
    v1_lag_sch = compute_h5_schmitt_lag_g4(
        v1, V1_BEAR_PROBA_COL, G4_EVENTS, TAU_EVAL,
    )
    v0s = summarize_h5(v0_lag)
    v1s = summarize_h5(v1_lag)
    v0s_g4 = summarize_h5(v0_lag_g4)
    v1s_g4 = summarize_h5(v1_lag_g4)
    v0s_sch = summarize_h5(v0_lag_sch)
    v1s_sch = summarize_h5(v1_lag_sch)

    verdict_argmax, reduction_pct_g4_argmax, reduction_pct_baseline_argmax = gate1_verdict(
        v0s_g4['median_lag'], v1s_g4['median_lag'], v1s['median_lag'],
    )
    verdict, reduction_pct_schmitt, v1_meets_floor = gate1_verdict_schmitt(
        v0s_sch['median_lag'], v1s_sch['median_lag'],
    )

    logger.info('=== Gate 1 summary (Amendment 6) ===')
    logger.info(f'Binding metric: Schmitt-tau-{TAU_EVAL:.2f} first-crossing on G4 drawdown events')
    logger.info(f'v0 median lag at tau={TAU_EVAL:.2f}: {v0s_sch["median_lag"]:.1f}d (fired {v0s_sch["n_fired"]}/{v0s_sch["n_onsets"]})')
    logger.info(f'v1 median lag at tau={TAU_EVAL:.2f}: {v1s_sch["median_lag"]:.1f}d (fired {v1s_sch["n_fired"]}/{v1s_sch["n_onsets"]})')
    logger.info(f'Reduction (v0 same-basis):          {reduction_pct_schmitt:.1f}%')
    logger.info(f'v1 meets <= 10d absolute floor:     {v1_meets_floor}')
    logger.info('')
    logger.info('Legacy argmax-at-0.5 (informational only post-Amendment 6):')
    logger.info(f'v0 median lag, G4 argmax: {v0s_g4["median_lag"]:.1f}d')
    logger.info(f'v1 median lag, G4 argmax: {v1s_g4["median_lag"]:.1f}d')
    logger.info(f'Reduction vs v0 G4-basis: {reduction_pct_g4_argmax:.1f}%')
    logger.info(f'Legacy argmax verdict:    {verdict_argmax}')
    logger.info('')
    logger.info(f'GATE 1 (binding, Amendment 6): {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
