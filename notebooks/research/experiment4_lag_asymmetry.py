"""Experiment 4 -- V12 lag-asymmetry decomposition.

Goal: decompose the +0.397 Sharpe gap between V12 one_day_lag and V12
near_close (both @ 5 bps) across regime-transition buckets:
  - BEAR-onset days       (regime[t-1] != BEAR and regime[t] == BEAR)
  - BEAR-exit days        (regime[t-1] == BEAR and regime[t] != BEAR)
  - Other-transition days (regime[t-1] != regime[t], neither BEAR-onset nor exit)
  - Persistent days       (regime[t-1] == regime[t])

If transition-day buckets explain > 80% of the daily P&L diff sum, the
asymmetry is TRANSITION-LOCALIZED and V12c readiness needs a stress test
at 10 bps in addition to the 7.5 bps already in place. If < 50%, the
asymmetry is DIFFUSE -- the lag tax is spread across calm days, so the
mechanism is elsewhere and needs deeper diagnosis. 50-80% is MIXED.

Data acquisition: we re-run the V12 harness twice (near_close + one_day_lag)
at 5 bps with the standard universe/dates. Per-day returns come directly
from the DailyRecord stream the engine already produces, so no harness
modification is needed -- only a thin orchestration wrapper.

Outputs:
  - diagnostics/regime/lag_asymmetry/decomposition.csv
  - diagnostics/regime/lag_asymmetry/verdict.txt
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import DailyRecord, run_variant
from src.research.ramp_phase4.variants import REGISTRY
from src.utils.logger import get_logger


logger = get_logger(__name__)


START = datetime(2017, 1, 1)
END = datetime(2026, 5, 16)
UNIVERSE = Path('config/universes/sp500-2025.csv')
INITIAL_CAPITAL = 100_000.0
COST_BPS = 5.0
VARIANT_ID = 'V12'
DELTA_REBALANCE_PCT = 0.02  # V12 default per ramp_phase4_v12_readiness orchestrator

V12_NC_SHARPE_REPORTED = 0.2683
V12_LAG_SHARPE_REPORTED = 0.6650
SHARPE_GAP_REPORTED = V12_LAG_SHARPE_REPORTED - V12_NC_SHARPE_REPORTED  # ~+0.397

LABELS_PATH = Path('diagnostics/regime/v0/labels.parquet')
OUT_DIR = Path('diagnostics/regime/lag_asymmetry')
DECOMP_CSV = OUT_DIR / 'decomposition.csv'
VERDICT_TXT = OUT_DIR / 'verdict.txt'

LOCALIZED_THRESHOLD = 0.80
DIFFUSE_THRESHOLD = 0.50


def _run_v12(timing_mode: str) -> List[DailyRecord]:
    cfg = HarnessConfig(
        start_date=START,
        end_date=END,
        universe_csv=UNIVERSE,
        initial_capital=INITIAL_CAPITAL,
        cost_bps_per_side=COST_BPS,
        timing_mode=timing_mode,
        delta_rebalance_pct=DELTA_REBALANCE_PCT,
    )
    spec = REGISTRY[VARIANT_ID]
    logger.info(f'[+] Running V12 timing_mode={timing_mode} cost={COST_BPS}bps...')
    t0 = time.time()
    records = run_variant(cfg, spec)
    elapsed = time.time() - t0
    if not records:
        raise RuntimeError(f'V12 {timing_mode} {COST_BPS}bps produced no records')
    logger.info(
        f'[+] V12 {timing_mode} done in {elapsed:.1f}s, n_days={len(records)}'
    )
    return records


def _records_to_returns(records: List[DailyRecord]) -> pd.Series:
    return pd.Series(
        data=[r.daily_return for r in records],
        index=pd.DatetimeIndex([pd.Timestamp(r.date) for r in records]),
        name='daily_return',
    )


def _annualized_sharpe(rets: pd.Series) -> float:
    rets = rets.dropna()
    if len(rets) < 2:
        return 0.0
    std = rets.std(ddof=1)
    if std == 0 or not np.isfinite(std) or std < 1e-15:
        return 0.0
    return float((rets.mean() * 252.0) / (std * np.sqrt(252.0)))


def _load_regime_labels() -> pd.DataFrame:
    df = pd.read_parquet(LABELS_PATH)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df[['regime']]


def _classify_days(regime: pd.Series) -> pd.DataFrame:
    prev = regime.shift(1)
    is_transition = (regime != prev) & prev.notna()
    is_bear_onset = (prev != 'BEAR') & (regime == 'BEAR') & prev.notna()
    is_bear_exit = (prev == 'BEAR') & (regime != 'BEAR') & prev.notna()
    is_other_transition = is_transition & (~is_bear_onset) & (~is_bear_exit)
    is_persistent = (regime == prev) & prev.notna()

    bucket = pd.Series('persistent', index=regime.index)
    bucket.loc[is_bear_onset] = 'bear_onset'
    bucket.loc[is_bear_exit] = 'bear_exit'
    bucket.loc[is_other_transition] = 'other_transition'
    # First day (prev is NaN) -> mark 'persistent' for accounting but it has no diff impact
    # since neither timing mode has a return on day 0 anyway.
    bucket.loc[prev.isna()] = 'persistent'

    return pd.DataFrame({
        'regime': regime,
        'prev_regime': prev,
        'bucket': bucket,
    })


def _decompose(daily_diff: pd.Series, buckets: pd.Series, sharpe_gap: float) -> pd.DataFrame:
    df = pd.DataFrame({'diff': daily_diff, 'bucket': buckets}).dropna(subset=['diff'])
    rows = []
    total_sum = df['diff'].sum()
    if total_sum == 0:
        logger.warning('[!] Total daily_diff sum is zero -- decomposition degenerate')
    for bucket_name in ('bear_onset', 'bear_exit', 'other_transition', 'persistent'):
        sub = df[df['bucket'] == bucket_name]
        n_days = int(len(sub))
        sum_diff = float(sub['diff'].sum())
        mean_diff = float(sub['diff'].mean()) if n_days else 0.0
        share = float(sum_diff / total_sum) if total_sum != 0 else 0.0
        sharpe_contrib = float(share * sharpe_gap)
        rows.append({
            'bucket': bucket_name,
            'n_days': n_days,
            'sum_daily_diff': sum_diff,
            'mean_daily_diff': mean_diff,
            'share_of_total': share,
            'implied_sharpe_contribution': sharpe_contrib,
        })
    return pd.DataFrame(rows)


def _verdict_text(decomp: pd.DataFrame, sharpe_gap: float, nc_sharpe: float,
                  lag_sharpe: float, transition_share: float, persistent_share: float,
                  by_bucket: Dict[str, Dict[str, float]]) -> str:
    if transition_share > LOCALIZED_THRESHOLD:
        verdict_tag = 'TRANSITION-LOCALIZED'
        interp = (
            f'Transition-day buckets explain {transition_share * 100:.1f}% of the daily '
            f'P&L difference between V12 one_day_lag and V12 near_close at 5 bps, which '
            f'exceeds the 80% TRANSITION-LOCALIZED threshold. The lag tax is concentrated '
            f'around regime-flip days -- specifically the days V12 executes a BEAR-onset '
            f'cash transition (or BEAR-exit re-entry) using the same-bar price the regime '
            f'panel just emitted. Implication: the V12c readiness cost-sensitivity gate '
            f'must add a 10 bps stress in addition to the 7.5 bps already in place, '
            f'because the 5-7.5 bps cost grid does not stress the transition-day '
            f'execution premium that the lag asymmetry exposes.'
        )
    elif transition_share < DIFFUSE_THRESHOLD:
        verdict_tag = 'DIFFUSE'
        interp = (
            f'Transition-day buckets explain only {transition_share * 100:.1f}% of the '
            f'daily P&L difference -- below the 50% DIFFUSE threshold. The lag tax is '
            f'spread across persistent (non-regime-flip) days, meaning the asymmetry is '
            f'not a "BEAR-transition cost" story but something more structural. '
            f'Implication: a deeper diagnosis is needed before any cash-transition '
            f'variant deploys. Candidate hypotheses to investigate: (a) the near_close '
            f'mode is paying a same-bar look-ahead penalty on every rebalance day (not '
            f'just transitions), (b) the regime panel has a systematic bias that affects '
            f'the held-position MTM differently between modes, (c) one_day_lag is '
            f'sidestepping intraday momentum reversion that near_close walks into daily.'
        )
    else:
        verdict_tag = 'MIXED'
        interp = (
            f'Transition-day buckets explain {transition_share * 100:.1f}% of the daily '
            f'P&L difference -- in the 50-80% MIXED band. Most of the asymmetry is '
            f'transition-driven but a non-trivial share leaks into persistent days. '
            f'Implication: analyst decision. The decomposition table below shows which '
            f'sub-bucket carries the leak; if persistent-day contribution is large in '
            f'absolute terms (rather than a small residual), that warrants the same '
            f'deeper-diagnosis path as DIFFUSE.'
        )

    def _pct(x: float) -> str:
        return f'{x * 100:+.1f}%' if abs(x) >= 0.001 else f'{x * 100:+.2f}%'

    lines = []
    lines.append('=== Experiment 4 Verdict ===')
    lines.append('')
    lines.append(f'VERDICT: {verdict_tag}')
    lines.append('')
    lines.append(
        f'Total Sharpe gap: {lag_sharpe - nc_sharpe:+.3f} '
        f'(V12 one_day_lag - V12 near_close at 5 bps)'
    )
    lines.append(f'  (Reference: orchestrator-reported gap = {SHARPE_GAP_REPORTED:+.3f})')
    lines.append(f'Transition-day share: {transition_share * 100:.1f}%')
    lines.append(f'Persistent-day share: {persistent_share * 100:.1f}%')
    lines.append('')
    lines.append('Decomposition:')
    for name in ('bear_onset', 'bear_exit', 'other_transition', 'persistent'):
        b = by_bucket[name]
        lines.append(
            f'  {name:<18s} share={b["share"] * 100:6.1f}%  '
            f'sharpe_contrib={b["sharpe"]:+.3f}  '
            f'(n_days={b["n_days"]})'
        )
    lines.append('')
    lines.append(interp)
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info('[+] Experiment 4: V12 lag-asymmetry decomposition')
    logger.info(
        f'[+] Data path: re-running V12 harness twice (no cached per-day artifacts found)'
    )

    nc_records = _run_v12('near_close')
    lag_records = _run_v12('one_day_lag')

    rets_nc = _records_to_returns(nc_records)
    rets_lag = _records_to_returns(lag_records)

    sharpe_nc = _annualized_sharpe(rets_nc)
    sharpe_lag = _annualized_sharpe(rets_lag)
    sharpe_gap = sharpe_lag - sharpe_nc
    logger.info(
        f'[+] Computed Sharpes: near_close={sharpe_nc:.4f}, '
        f'one_day_lag={sharpe_lag:.4f}, gap={sharpe_gap:+.4f}'
    )
    logger.info(
        f'[+] Reference Sharpes from orchestrator: '
        f'nc={V12_NC_SHARPE_REPORTED:.4f}, lag={V12_LAG_SHARPE_REPORTED:.4f}, '
        f'gap={SHARPE_GAP_REPORTED:+.4f}'
    )

    # Align series on common dates so the diff is well-defined.
    common = rets_nc.index.intersection(rets_lag.index)
    rets_nc = rets_nc.loc[common]
    rets_lag = rets_lag.loc[common]
    daily_diff = rets_lag - rets_nc
    logger.info(
        f'[+] daily_diff: n={len(daily_diff)}, sum={daily_diff.sum():+.6f}, '
        f'mean={daily_diff.mean():+.6f}'
    )

    labels = _load_regime_labels()
    # Anchor regimes to harness dates. Inner-join: only use days present in both.
    aligned = pd.DataFrame({'daily_diff': daily_diff}).join(labels, how='left')
    if aligned['regime'].isna().any():
        missing = int(aligned['regime'].isna().sum())
        logger.warning(
            f'[!] {missing} harness days have no regime label -- dropping them from buckets'
        )
        aligned = aligned.dropna(subset=['regime'])

    classes = _classify_days(aligned['regime'])
    decomp = _decompose(
        daily_diff=aligned['daily_diff'],
        buckets=classes['bucket'],
        sharpe_gap=sharpe_gap,
    )

    # Localization metric: share of daily_diff sum from any-transition bucket.
    transition_buckets = {'bear_onset', 'bear_exit', 'other_transition'}
    transition_share = float(
        decomp.loc[decomp['bucket'].isin(transition_buckets), 'share_of_total'].sum()
    )
    persistent_share = float(
        decomp.loc[decomp['bucket'] == 'persistent', 'share_of_total'].sum()
    )

    decomp.to_csv(DECOMP_CSV, index=False)
    logger.info(f'[+] Wrote {DECOMP_CSV}')

    by_bucket = {
        row['bucket']: {
            'share': float(row['share_of_total']),
            'sharpe': float(row['implied_sharpe_contribution']),
            'n_days': int(row['n_days']),
        }
        for _, row in decomp.iterrows()
    }
    verdict = _verdict_text(
        decomp=decomp,
        sharpe_gap=sharpe_gap,
        nc_sharpe=sharpe_nc,
        lag_sharpe=sharpe_lag,
        transition_share=transition_share,
        persistent_share=persistent_share,
        by_bucket=by_bucket,
    )
    VERDICT_TXT.write_text(verdict, encoding='utf-8')
    logger.info(f'[+] Wrote {VERDICT_TXT}')

    # Echo the verdict header to the log so the orchestrator can grep it.
    logger.info('[+] Decomposition complete.')
    for line in verdict.splitlines()[:14]:
        logger.info(f'    {line}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
