"""Experiment 5 -- OMR cross-check on regime-detector failure modes.

Question:
  Does OMR's per-trade P&L by regime exhibit the same flicker/lag fingerprint
  the regime detector showed when consumed by RAMP (V12 BEAR-to-cash)?

  - If YES (DETECTOR-WIDE) -- detector improvement (WS-3a/WS-3b/WS-3c) scales
    across both strategies. WS-3 leapfrogs V12c in portfolio priority.
  - If NO (RAMP-SPECIFIC) -- V12c is the higher-leverage track. WS-3 payoff is
    RAMP-only.
  - AMBIGUOUS otherwise.

Method:
  1. Load `output/backtests/omr_original_universe_2017_2024_trades.csv`
     (2335 trades, 2018-2024). The `regime` column was populated by the
     production detector at trade entry.
  2. Per-regime trade-level Sharpe = mean(net_return) / std(net_return).
     Bootstrap 95% CI when n > 50.
  3. Transition-day vs persistent-day decomposition using
     `diagnostics/regime/v0/labels.parquet` for day-over-day transition tags.
  4. BEAR-onset alignment: for each BEAR-onset (day-over-day transition into
     BEAR) inside the OMR window, average OMR net_return in [-3, +3] trading
     days. Compare against overall OMR mean.
  5. Apply Sharpe-range / max-Sharpe > 30% decision criterion vs RAMP V12
     near_close (Sharpe range across regimes is comparable per V12 readiness
     panel).

Outputs:
  diagnostics/omr_cross_check/omr_per_regime_sharpe.csv
  diagnostics/omr_cross_check/omr_transition_vs_persistent.csv
  diagnostics/omr_cross_check/omr_bear_onset_alignment.csv
  diagnostics/omr_cross_check/verdict.txt

Run:
    PYTHONPATH=. python notebooks/research/experiment5_omr_regime_attribution.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import logger


TRADES_PATH = Path('output/backtests/omr_original_universe_2017_2024_trades.csv')
LABELS_PATH = Path('diagnostics/regime/v0/labels.parquet')
OUT_DIR = Path('diagnostics/omr_cross_check')

PER_REGIME_CSV = OUT_DIR / 'omr_per_regime_sharpe.csv'
TRANSITION_CSV = OUT_DIR / 'omr_transition_vs_persistent.csv'
BEAR_ONSET_CSV = OUT_DIR / 'omr_bear_onset_alignment.csv'
VERDICT_TXT = OUT_DIR / 'verdict.txt'

CANONICAL_REGIMES = ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']
BOOTSTRAP_N = 1000
BOOTSTRAP_MIN_N = 50
BEAR_WINDOW_DAYS = 3
DECISION_THRESHOLD_PCT = 0.30
RNG_SEED = 42


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_PATH)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('entry_date').reset_index(drop=True)
    logger.info(f'[+] Loaded {len(df)} OMR trades from {TRADES_PATH}')
    logger.info(
        f'[+] Trade window: {df["entry_date"].min().date()} -> '
        f'{df["entry_date"].max().date()}'
    )
    counts = df['regime'].value_counts()
    logger.info(f'[+] OMR regime taxonomy (entry-time): {dict(counts)}')
    return df


def load_labels() -> pd.DataFrame:
    df = pd.read_parquet(LABELS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['prev_regime'] = df['regime'].shift(1)
    df['is_transition'] = (df['regime'] != df['prev_regime']) & df['prev_regime'].notna()
    df['is_bear_onset'] = (df['regime'] == 'BEAR') & (df['prev_regime'] != 'BEAR') & df['prev_regime'].notna()
    logger.info(
        f'[+] Loaded {len(df)} regime labels {df["date"].min().date()} -> '
        f'{df["date"].max().date()}'
    )
    return df


def bootstrap_sharpe_ci(
    returns: np.ndarray, n_resamples: int = BOOTSTRAP_N, seed: int = RNG_SEED
) -> Tuple[float, float]:
    if len(returns) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    n = len(returns)
    sharpes = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = returns[rng.integers(0, n, size=n)]
        s = sample.std(ddof=1)
        sharpes[i] = sample.mean() / s if s > 0 else 0.0
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return float(lo), float(hi)


def per_trade_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float('nan')
    s = returns.std(ddof=1)
    if s <= 0:
        return float('nan')
    return float(returns.mean() / s)


def per_regime_sharpe(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for regime in CANONICAL_REGIMES:
        subset = trades[trades['regime'] == regime]
        n = len(subset)
        if n == 0:
            rows.append(
                dict(
                    regime=regime,
                    n_trades=0,
                    mean_net_return=float('nan'),
                    std_net_return=float('nan'),
                    sharpe_per_trade=float('nan'),
                    sharpe_ci_low=float('nan'),
                    sharpe_ci_high=float('nan'),
                )
            )
            continue
        returns = subset['net_return'].to_numpy(dtype=float)
        mean_r = float(returns.mean())
        std_r = float(returns.std(ddof=1)) if n > 1 else float('nan')
        sharpe = per_trade_sharpe(returns)
        if n >= BOOTSTRAP_MIN_N:
            ci_lo, ci_hi = bootstrap_sharpe_ci(returns)
        else:
            ci_lo = float('nan')
            ci_hi = float('nan')
        rows.append(
            dict(
                regime=regime,
                n_trades=n,
                mean_net_return=mean_r,
                std_net_return=std_r,
                sharpe_per_trade=sharpe,
                sharpe_ci_low=ci_lo,
                sharpe_ci_high=ci_hi,
            )
        )
    df = pd.DataFrame(rows)
    logger.info('[+] Per-regime Sharpe computed')
    return df


def transition_vs_persistent(
    trades: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    label_lookup = labels[['date', 'is_transition', 'regime']].rename(
        columns={'date': 'entry_date', 'regime': 'label_regime'}
    )
    merged = trades.merge(label_lookup, on='entry_date', how='left')
    matched = int(merged['is_transition'].notna().sum())
    logger.info(
        f'[+] Joined {matched}/{len(merged)} OMR trades to regime labels by entry_date'
    )

    rows: List[Dict] = []
    for bucket_name, mask in [
        ('transition_day', merged['is_transition'] == True),
        ('persistent_day', merged['is_transition'] == False),
    ]:
        subset = merged[mask]
        n = len(subset)
        if n == 0:
            rows.append(
                dict(
                    bucket=bucket_name,
                    n_trades=0,
                    mean_net_return=float('nan'),
                    std_net_return=float('nan'),
                    sharpe_per_trade=float('nan'),
                )
            )
            continue
        returns = subset['net_return'].to_numpy(dtype=float)
        rows.append(
            dict(
                bucket=bucket_name,
                n_trades=n,
                mean_net_return=float(returns.mean()),
                std_net_return=float(returns.std(ddof=1)) if n > 1 else float('nan'),
                sharpe_per_trade=per_trade_sharpe(returns),
            )
        )

    overall = merged['net_return'].to_numpy(dtype=float)
    rows.append(
        dict(
            bucket='overall',
            n_trades=len(overall),
            mean_net_return=float(overall.mean()),
            std_net_return=float(overall.std(ddof=1)),
            sharpe_per_trade=per_trade_sharpe(overall),
        )
    )
    df = pd.DataFrame(rows)
    logger.info('[+] Transition vs persistent decomposition computed')
    return df


def bear_onset_alignment(
    trades: pd.DataFrame, labels: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    bear_onsets_all = labels[labels['is_bear_onset']].copy()
    omr_start = trades['entry_date'].min()
    omr_end = trades['entry_date'].max()

    trading_days = labels['date'].to_numpy()

    rows: List[Dict] = []
    out_of_window = 0
    in_window_n = 0
    for _, ev in bear_onsets_all.iterrows():
        onset_date = ev['date']
        if onset_date < omr_start or onset_date > omr_end:
            out_of_window += 1
            continue
        in_window_n += 1
        idx = np.searchsorted(trading_days, onset_date.to_datetime64())
        lo_idx = max(0, idx - BEAR_WINDOW_DAYS)
        hi_idx = min(len(trading_days) - 1, idx + BEAR_WINDOW_DAYS)
        win_start = pd.Timestamp(trading_days[lo_idx])
        win_end = pd.Timestamp(trading_days[hi_idx])
        in_window = trades[
            (trades['entry_date'] >= win_start) & (trades['entry_date'] <= win_end)
        ]
        if len(in_window) > 0:
            mean_r = float(in_window['net_return'].mean())
        else:
            mean_r = float('nan')
        rows.append(
            dict(
                onset_date=onset_date.date().isoformat(),
                window_start=win_start.date().isoformat(),
                window_end=win_end.date().isoformat(),
                omr_trade_count_window=len(in_window),
                mean_net_return_window=mean_r,
            )
        )

    df = pd.DataFrame(rows)
    overall_mean = float(trades['net_return'].mean())
    df['mean_net_return_overall'] = overall_mean

    in_window_returns_total: List[float] = []
    for _, row in df.iterrows():
        if row['omr_trade_count_window'] == 0 or pd.isna(row['mean_net_return_window']):
            continue
        win_start = pd.Timestamp(row['window_start'])
        win_end = pd.Timestamp(row['window_end'])
        slc = trades[
            (trades['entry_date'] >= win_start) & (trades['entry_date'] <= win_end)
        ]
        in_window_returns_total.extend(slc['net_return'].tolist())

    if in_window_returns_total:
        pooled_mean = float(np.mean(in_window_returns_total))
        pooled_n = len(in_window_returns_total)
    else:
        pooled_mean = float('nan')
        pooled_n = 0

    summary = dict(
        bear_onsets_total=int(len(bear_onsets_all)),
        bear_onsets_in_omr_window=in_window_n,
        bear_onsets_out_of_window=out_of_window,
        missing_fraction=float(out_of_window) / max(1, int(len(bear_onsets_all))),
        bear_onsets_with_omr_trade=int((df['omr_trade_count_window'] > 0).sum()),
        pooled_window_mean_return=pooled_mean,
        pooled_window_trade_count=pooled_n,
        overall_mean_return=overall_mean,
    )

    logger.info(
        f'[+] BEAR onsets: {summary["bear_onsets_total"]} total, '
        f'{summary["bear_onsets_in_omr_window"]} in OMR window, '
        f'{summary["bear_onsets_out_of_window"]} outside '
        f'(missing fraction {summary["missing_fraction"]:.3f})'
    )
    logger.info(
        f'[+] BEAR-onset window pooled mean return: {pooled_mean:.6f} '
        f'(n={pooled_n}) vs overall {overall_mean:.6f}'
    )
    return df, summary


def build_verdict(
    per_regime: pd.DataFrame,
    transition: pd.DataFrame,
    bear_summary: Dict,
) -> Tuple[str, Dict]:
    valid = per_regime[per_regime['n_trades'] > 0].copy()
    sharpes = valid['sharpe_per_trade'].to_numpy(dtype=float)
    sharpes_finite = sharpes[np.isfinite(sharpes)]
    if len(sharpes_finite) >= 2:
        max_abs = float(np.max(np.abs(sharpes_finite)))
        rng = float(sharpes_finite.max() - sharpes_finite.min())
        ratio = rng / max_abs if max_abs > 0 else float('nan')
    else:
        max_abs = float('nan')
        rng = float('nan')
        ratio = float('nan')

    trans_row = transition[transition['bucket'] == 'transition_day'].iloc[0]
    pers_row = transition[transition['bucket'] == 'persistent_day'].iloc[0]
    trans_sharpe = float(trans_row['sharpe_per_trade'])
    pers_sharpe = float(pers_row['sharpe_per_trade'])
    trans_gap = trans_sharpe - pers_sharpe

    regimes_observed = int((per_regime['n_trades'] > 0).sum())
    bear_n = int(per_regime[per_regime['regime'] == 'BEAR']['n_trades'].iloc[0])
    unpred_n = int(per_regime[per_regime['regime'] == 'UNPREDICTABLE']['n_trades'].iloc[0])

    if regimes_observed <= 3 and (bear_n + unpred_n) < 20:
        verdict = 'AMBIGUOUS'
        rationale = (
            'OMR adapter filters BEAR/UNPREDICTABLE entries via its Bayesian-bucket '
            'screen, so the regime column carries only STRONG_BULL/WEAK_BULL/SIDEWAYS. '
            'A direct Sharpe-by-regime range comparison vs RAMP cannot resolve '
            'DETECTOR-WIDE vs RAMP-SPECIFIC at trade granularity: OMR\'s defence '
            'against BEAR is structural (no trade), not a Sharpe degradation we can '
            'compare to RAMP\'s BEAR-day equity exposure.'
        )
    elif np.isfinite(ratio) and ratio >= DECISION_THRESHOLD_PCT:
        verdict = 'DETECTOR-WIDE'
        rationale = (
            f'OMR Sharpe-by-regime range/max = {ratio:.1%} >= {DECISION_THRESHOLD_PCT:.0%}, '
            'comparable to RAMP V12\'s regime-conditional swing. Detector improvement '
            '(WS-3a/WS-3b/WS-3c) scales across both strategies.'
        )
    else:
        verdict = 'RAMP-SPECIFIC'
        rationale = (
            f'OMR Sharpe-by-regime range/max = {ratio:.1%} < {DECISION_THRESHOLD_PCT:.0%}. '
            'OMR is robust to the detector\'s flicker/lag failures (the Bayesian '
            'probability bucket dampens regime signal). V12c (single-strategy RAMP '
            'fix) is higher leverage than WS-3.'
        )

    lines = ['=== Experiment 5 Verdict ===', '', f'VERDICT: {verdict}', '']
    lines.append('OMR per-regime Sharpe (per-trade):')
    for _, r in per_regime.iterrows():
        if r['n_trades'] == 0:
            lines.append(
                f'  {r["regime"]:<14} (n=0)  -- absent from OMR trade log'
            )
        else:
            lines.append(
                f'  {r["regime"]:<14} {r["sharpe_per_trade"]:>6.3f} '
                f'(n={int(r["n_trades"])})'
            )

    if np.isfinite(ratio):
        lines.append('')
        lines.append(f'Sharpe range / max(|Sharpe|): {ratio:.1%}')
        lines.append(f'  (range={rng:.3f}, max|Sharpe|={max_abs:.3f})')

    lines.append('')
    lines.append(
        f'Transition-day Sharpe: {trans_sharpe:>6.3f} '
        f'(n={int(trans_row["n_trades"])})'
    )
    lines.append(
        f'Persistent-day Sharpe: {pers_sharpe:>6.3f} '
        f'(n={int(pers_row["n_trades"])})'
    )
    lines.append(f'Transition-day gap (transition - persistent): {trans_gap:+.3f}')

    lines.append('')
    lines.append(
        f'BEAR-onset window OMR mean return: '
        f'{bear_summary["pooled_window_mean_return"]:.6f} '
        f'(n={bear_summary["pooled_window_trade_count"]})'
    )
    lines.append(
        f'Overall OMR mean return:           '
        f'{bear_summary["overall_mean_return"]:.6f}'
    )
    lines.append(
        f'BEAR onsets out of OMR window: {bear_summary["bear_onsets_out_of_window"]} '
        f'/ {bear_summary["bear_onsets_total"]} '
        f'(missing fraction {bear_summary["missing_fraction"]:.3f})'
    )

    lines.append('')
    lines.append('Interpretation:')
    lines.append(rationale)
    if verdict == 'AMBIGUOUS':
        lines.append('')
        lines.append(
            'Analyst guidance: treat WS-3 (detector improvement) as RAMP-attributable '
            'unless and until a separate study instruments OMR\'s Bayesian-bucket fill '
            'to expose per-day regime sensitivity. At trade granularity, OMR shows no '
            'BEAR/UNPREDICTABLE exposure to compare with.'
        )

    metrics = dict(
        verdict=verdict,
        ratio=ratio,
        sharpe_range=rng,
        max_abs_sharpe=max_abs,
        transition_sharpe=trans_sharpe,
        persistent_sharpe=pers_sharpe,
        transition_gap=trans_gap,
    )

    return '\n'.join(lines) + '\n', metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trades = load_trades()
    labels = load_labels()

    per_regime = per_regime_sharpe(trades)
    per_regime.to_csv(PER_REGIME_CSV, index=False)
    logger.info(f'[+] Wrote {PER_REGIME_CSV}')

    transition = transition_vs_persistent(trades, labels)
    transition.to_csv(TRANSITION_CSV, index=False)
    logger.info(f'[+] Wrote {TRANSITION_CSV}')

    bear_df, bear_summary = bear_onset_alignment(trades, labels)
    bear_df.to_csv(BEAR_ONSET_CSV, index=False)
    logger.info(f'[+] Wrote {BEAR_ONSET_CSV}')

    verdict_text, metrics = build_verdict(per_regime, transition, bear_summary)
    VERDICT_TXT.write_text(verdict_text, encoding='utf-8')
    logger.info(f'[+] Wrote {VERDICT_TXT}')
    logger.info(f'[+] Verdict: {metrics["verdict"]}')


if __name__ == '__main__':
    main()
