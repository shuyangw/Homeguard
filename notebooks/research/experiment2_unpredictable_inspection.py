"""Experiment 2 -- UNPREDICTABLE hand-inspection (V12-up-cash fragility test).

V12-up-cash (UNPREDICTABLE='cash' sensitivity variant from V12 readiness) produced
Sharpe 0.586 at 5 bps near_close vs V11's 0.528 -- a +0.058 edge. UNPREDICTABLE
fires only ~1.7% of trading days (41 days across 14 runs in 2017-2026). This
script tests whether the +0.058 Sharpe edge is concentrated in 1-3 specific
events (FRAGILE) or spread evenly across the 14 firings (ROBUST).

Methodology:
  1. Identify UNPREDICTABLE events as maximal contiguous runs of UNPREDICTABLE
     daily labels in `diagnostics/regime/v0/labels.parquet`.
  2. For each event, compute SPY return during the event window from
     `diagnostics/data/spy_vix_2016_2026.parquet`, plus forward 5/10/20d SPY
     returns post-event.
  3. Under V12-up-cash the strategy is in CASH (0% SPY) during UNPREDICTABLE,
     so the V12-up-cash "avoided loss attribution" per event = -1 * SPY
     return during the event. Negative SPY return -> positive avoided loss.
  4. Concentration metric: top-3 events' absolute attribution as fraction of
     total absolute attribution.
  5. Sharpe contribution proxy: linear scaling of the +0.058 Sharpe delta by
     each event's attribution share. **First-order approximation** -- assumes
     constant per-event volatility.

Decision criterion:
  - FRAGILE   if top-3 attribution share > 75%  -> do NOT run E6 (V12c readiness)
  - ROBUST    if top-3 attribution share < 50%  -> proceed to E6
  - AMBIGUOUS if 50%-75%                        -> analyst decides

Inputs:
  diagnostics/regime/v0/labels.parquet          (regime detector v0 labels)
  diagnostics/data/spy_vix_2016_2026.parquet    (SPY+VIX panel)

Outputs:
  diagnostics/regime/unpredictable_events/per_event.csv
  diagnostics/regime/unpredictable_events/verdict.txt

Run:
    python notebooks/research/experiment2_unpredictable_inspection.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import logger


LABELS_PATH = Path('diagnostics/regime/v0/labels.parquet')
PANEL_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
OUT_DIR = Path('diagnostics/regime/unpredictable_events')

V11_SHARPE = 0.528
V12_UP_CASH_SHARPE = 0.586
SHARPE_DELTA = V12_UP_CASH_SHARPE - V11_SHARPE  # +0.058

CONCENTRATION_FRAGILE = 0.75
CONCENTRATION_ROBUST = 0.50


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading regime labels from {LABELS_PATH}")
    labels = pd.read_parquet(LABELS_PATH).sort_values('date').reset_index(drop=True)
    logger.info(f"  labels shape={labels.shape} date_range={labels['date'].min().date()}..{labels['date'].max().date()}")

    logger.info(f"Loading SPY+VIX panel from {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH).sort_index()
    logger.info(f"  panel shape={panel.shape} date_range={panel.index.min().date()}..{panel.index.max().date()}")
    return labels, panel


def identify_events(labels: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame, one row per maximal contiguous UNPREDICTABLE run."""
    is_unp = (labels['regime'] == 'UNPREDICTABLE').values
    # Build run-id only along the UNPREDICTABLE rows.
    # A "new run" starts wherever the current row is UNPREDICTABLE but the
    # previous regime label was something else (or this is the first row).
    prev_is_unp = np.concatenate([[False], is_unp[:-1]])
    new_run = is_unp & ~prev_is_unp
    run_idx = np.cumsum(new_run)
    # run_idx is only meaningful where is_unp is True
    df = labels[['date', 'regime']].copy()
    df['run_idx'] = run_idx
    df = df[is_unp]

    events = (
        df.groupby('run_idx')
        .agg(start_date=('date', 'min'), end_date=('date', 'max'), n_days=('date', 'count'))
        .reset_index(drop=True)
    )
    events.insert(0, 'event_id', np.arange(1, len(events) + 1))
    logger.info(f"  identified {len(events)} UNPREDICTABLE events totaling {events['n_days'].sum()} days")
    logger.info(f"  run length stats: mean={events['n_days'].mean():.2f} median={events['n_days'].median():.1f} max={events['n_days'].max()}")
    return events


def forward_close(panel: pd.DataFrame, anchor_date: pd.Timestamp, n_trading_days: int) -> float | None:
    """Return SPY close `n_trading_days` trading days after anchor_date, or None."""
    idx = panel.index
    pos = idx.searchsorted(anchor_date)
    # We want the close at position `pos + n_trading_days` where pos is the
    # position of anchor_date itself (which we assume is a trading day).
    if pos >= len(idx) or idx[pos] != anchor_date:
        # anchor not in index, fall back to nearest prior
        pos = pos - 1
        if pos < 0:
            return None
    target_pos = pos + n_trading_days
    if target_pos >= len(idx):
        return None
    return float(panel['spy_close'].iloc[target_pos])


def compute_per_event_table(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Build per-event metrics: SPY return during event, forward returns,
    avoided-loss attribution under V12-up-cash assumption."""
    rows = []
    for _, ev in events.iterrows():
        start = ev['start_date']
        end = ev['end_date']
        # SPY close on (start - 1 trading day) -> close on end. Event return
        # = close[end] / close[day_before_start] - 1. This is the simple return
        # that V12-up-cash forgoes by being in cash from start through end.
        idx = panel.index
        start_pos = idx.searchsorted(start)
        if start_pos >= len(idx) or idx[start_pos] != start:
            logger.warning(f"  event {ev['event_id']}: start={start.date()} not in panel; skipping")
            continue
        end_pos = idx.searchsorted(end)
        if end_pos >= len(idx) or idx[end_pos] != end:
            logger.warning(f"  event {ev['event_id']}: end={end.date()} not in panel; skipping")
            continue
        if start_pos == 0:
            logger.warning(f"  event {ev['event_id']}: no prior close before {start.date()}; skipping")
            continue

        close_before_start = float(panel['spy_close'].iloc[start_pos - 1])
        close_end = float(panel['spy_close'].iloc[end_pos])
        spy_return_during = close_end / close_before_start - 1.0

        fwd5 = forward_close(panel, end, 5)
        fwd10 = forward_close(panel, end, 10)
        fwd20 = forward_close(panel, end, 20)
        spy_return_forward_5d = (fwd5 / close_end - 1.0) if fwd5 is not None else np.nan
        spy_return_forward_10d = (fwd10 / close_end - 1.0) if fwd10 is not None else np.nan
        spy_return_forward_20d = (fwd20 / close_end - 1.0) if fwd20 is not None else np.nan

        # V12-up-cash avoided-loss attribution: under V12-up-cash, strategy is
        # in CASH (0% SPY) during UNPREDICTABLE. V11 would have been in equity
        # (per V11 logic, approx fully invested in SPY). So V12-up-cash forgoes
        # the SPY return during the event. avoided_loss = -spy_return_during.
        # Positive avoided_loss = SPY went down, cash avoided the loss.
        # Negative avoided_loss = SPY went up, cash missed the gain.
        avoided_loss = -spy_return_during

        rows.append(
            {
                'event_id': int(ev['event_id']),
                'start_date': start.date().isoformat(),
                'end_date': end.date().isoformat(),
                'n_days': int(ev['n_days']),
                'spy_return_during': spy_return_during,
                'spy_return_forward_5d': spy_return_forward_5d,
                'spy_return_forward_10d': spy_return_forward_10d,
                'spy_return_forward_20d': spy_return_forward_20d,
                'avoided_loss_attribution': avoided_loss,
            }
        )

    per_event = pd.DataFrame(rows)
    per_event['abs_attribution'] = per_event['avoided_loss_attribution'].abs()
    per_event = per_event.sort_values('abs_attribution', ascending=False).reset_index(drop=True)
    return per_event


def compute_verdict(per_event: pd.DataFrame) -> dict:
    """Apply concentration decision criterion."""
    total_abs = per_event['abs_attribution'].sum()
    if total_abs == 0:
        share = 0.0
    else:
        share = per_event['abs_attribution'].iloc[:3].sum() / total_abs
    top3 = per_event.head(3)
    sharpe_contribs = SHARPE_DELTA * (per_event['abs_attribution'] / total_abs)
    top3_sharpe = float(sharpe_contribs.iloc[:3].sum())

    if share > CONCENTRATION_FRAGILE:
        verdict = 'FRAGILE'
    elif share < CONCENTRATION_ROBUST:
        verdict = 'ROBUST'
    else:
        verdict = 'AMBIGUOUS'

    return {
        'verdict': verdict,
        'top3_share': share,
        'top3_sharpe_contribution': top3_sharpe,
        'top3': top3,
        'total_abs_attribution': total_abs,
        'sharpe_contribs': sharpe_contribs,
    }


def write_verdict(result: dict, n_events: int) -> Path:
    out = OUT_DIR / 'verdict.txt'
    top3 = result['top3']
    top3_lines = '\n'.join(
        f"  event_id={r.event_id} start_date={r.start_date} end_date={r.end_date} "
        f"n_days={r.n_days} spy_return_during={r.spy_return_during:+.2%} "
        f"avoided_loss={r.avoided_loss_attribution:+.2%}"
        for r in top3.itertuples(index=False)
    )

    verdict = result['verdict']
    if verdict == 'FRAGILE':
        interp = (
            "Top-3 events drive more than 75% of the absolute avoided-loss attribution "
            "under V12-up-cash. The Sharpe edge over V11 is dominated by a handful of "
            "specific historical events (e.g. COVID crash, tariff selloff). V12c is "
            "fragile -- do NOT run Experiment 6 (V12c formal readiness). The +0.058 "
            "Sharpe is not a statistically robust improvement."
        )
    elif verdict == 'ROBUST':
        interp = (
            "Top-3 events drive less than 50% of the absolute avoided-loss attribution "
            "under V12-up-cash. The Sharpe edge over V11 is reasonably distributed across "
            "the UNPREDICTABLE firings. V12c is structurally robust -- proceed to "
            "Experiment 6 (V12c formal readiness)."
        )
    else:
        interp = (
            "Top-3 events drive between 50% and 75% of the absolute avoided-loss "
            "attribution under V12-up-cash. Concentration is moderate. Decision is "
            "ambiguous -- defer to analyst review. Consider per-day P&L re-run before "
            "committing to Experiment 6."
        )

    body = f"""=== Experiment 2 Verdict ===

VERDICT: {verdict}

N events: {n_events}
Top-3 attribution share: {result['top3_share']:.1%}
Top-3 implied Sharpe contribution: {result['top3_sharpe_contribution']:+.4f} (out of {SHARPE_DELTA:+.4f} V12-up-cash vs V11 delta)
Total absolute attribution: {result['total_abs_attribution']:+.4f}

Top-3 events (by |avoided_loss|):
{top3_lines}

Decision criterion:
  FRAGILE   if top-3 share > {CONCENTRATION_FRAGILE:.0%}  -> do NOT run E6
  ROBUST    if top-3 share < {CONCENTRATION_ROBUST:.0%}  -> proceed to E6
  AMBIGUOUS if {CONCENTRATION_ROBUST:.0%}-{CONCENTRATION_FRAGILE:.0%}                  -> analyst decides

Interpretation:
{interp}

Note: Sharpe contribution is a first-order linear-scaling proxy. It assumes
constant volatility across events. A more rigorous version would require
re-running V12-up-cash with per-day P&L logging and computing the actual
Sharpe difference contribution.

Sign convention: avoided_loss = -spy_return_during.
  Positive avoided_loss -> SPY went DOWN during the event; cash avoided the loss.
  Negative avoided_loss -> SPY went UP during the event;   cash missed the gain.
"""
    out.write_text(body)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, panel = load_inputs()

    logger.info("Identifying UNPREDICTABLE events ...")
    events = identify_events(labels)

    logger.info("Computing per-event SPY returns and avoided-loss attribution ...")
    per_event = compute_per_event_table(events, panel)
    if per_event.empty:
        logger.error("[-] per_event table is empty; aborting")
        return 1

    out_csv = OUT_DIR / 'per_event.csv'
    per_event_out = per_event.drop(columns=['abs_attribution'])
    per_event_out.to_csv(out_csv, index=False, float_format='%.6f')
    logger.info(f"[+] wrote per-event table -> {out_csv} (rows={len(per_event_out)})")

    logger.info("Computing concentration / verdict ...")
    result = compute_verdict(per_event)
    logger.info(f"  verdict={result['verdict']}  top-3 share={result['top3_share']:.1%}  top-3 Sharpe contribution={result['top3_sharpe_contribution']:+.4f}")

    verdict_path = write_verdict(result, n_events=len(per_event))
    logger.info(f"[+] wrote verdict -> {verdict_path}")

    # Summary log block
    logger.info("--- Top-3 events ---")
    for r in result['top3'].itertuples(index=False):
        logger.info(
            f"  event_id={r.event_id} {r.start_date}..{r.end_date} "
            f"n_days={r.n_days} spy_return_during={r.spy_return_during:+.2%} "
            f"avoided_loss={r.avoided_loss_attribution:+.2%}"
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
