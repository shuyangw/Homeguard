#!/usr/bin/env python3
"""Backfill hg_strategy_realized_pnl_lifetime_usd and hg_strategy_equity_usd
into VictoriaMetrics from trade logs.

Reads all trades_*.jsonl in the log dir chronologically, computes per-strategy
cumulative realized PnL at each exit timestamp, and POSTs as timestamped
Prometheus-format gauge values to VM's /api/v1/import/prometheus endpoint.

For strategies whose live equity gauge formula was changed mid-flight
(commit ee6d635 added lifetime_realized to RAMP's equity), also backfills
hg_strategy_equity_usd = initial_capital + lifetime_realized at each exit
timestamp. This smooths the visible step in the Drawdown panel that arises
from comparing post-fix equity against pre-fix historical samples.

Idempotent: re-running with the same input emits the same (metric, labels,
timestamp, value) tuples. VM dedupes on (series, timestamp) so re-imports are
safe.

When to run:
- After deploying lifetime-PnL or the corrected equity formula, to backfill
  trade-log history that pre-dates the gauges' live emission.
- After a VM data-loss event, to restore historical points (limited by VM's
  retention period, default 90d in our setup).

Note: VM silently drops imports older than its retention window. Today
that's ~90d, so trades older than ~3 months won't appear in queries.

Equity caveat: backfill omits the unrealized component (we don't have
historical position values). At an exit timestamp the just-closed position
has 0 unrealized; other concurrently-held positions have non-zero unrealized
that we can't reconstruct. RAMP's full-portfolio unrealized has stayed
within ~+/-$3K historically -- small relative to realized swings, acceptable
for cosmetic dashboard purposes (smooths Drawdown), not accounting-grade.

Labels match what live emission produces (per scrape_config in
/etc/homeguard/scrape.yaml):
  ramp -> {instance="127.0.0.1:8082", job="homeguard-ramp", strategy="ramp"}
  cscm -> {instance="127.0.0.1:8084", job="homeguard-cscm", strategy="cscm"}
  omr  -> {instance="127.0.0.1:8081", job="homeguard-omr",  strategy="omr"}
  mp   -> {instance="127.0.0.1:8083", job="homeguard-mp",   strategy="mp"}

Usage on EC2:
  sudo -u ec2-user python3 scripts/ops/backfill_lifetime_pnl.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import urllib.request

LOG_DIR = Path('/home/ec2-user/logs')
VM_URL = 'http://127.0.0.1:8428/api/v1/import/prometheus'
METRIC = 'hg_strategy_realized_pnl_lifetime_usd'
EQUITY_METRIC = 'hg_strategy_equity_usd'

LABELS = {
    'ramp': '{instance="127.0.0.1:8082",job="homeguard-ramp",strategy="ramp"}',
    'cscm': '{instance="127.0.0.1:8084",job="homeguard-cscm",strategy="cscm"}',
    'omr':  '{instance="127.0.0.1:8081",job="homeguard-omr",strategy="omr"}',
    'mp':   '{instance="127.0.0.1:8083",job="homeguard-mp",strategy="mp"}',
}

# Strategies whose equity gauge was affected by the formula bug fixed in
# commit ee6d635. CSCM uses broker.portfolio_value directly so its equity
# series in VM was always correct; OMR/MP aren't currently scraped.
EQUITY_BACKFILL_STRATEGIES = {'ramp'}
EQUITY_INITIAL_CAPITAL_USD = {
    'ramp': 100000.0,
}


def main() -> int:
    all_exits = []
    # Earliest trade-of-any-kind timestamp per strategy. Used to emit a $0
    # baseline for strategies that have entries in the trade log but no
    # exits yet (e.g. CSCM in early operation), so the chart shows a flat
    # zero line across the visible range instead of a single live dot.
    strategy_earliest_trade_ts: dict = {}
    for f in sorted(LOG_DIR.glob('trades_*.jsonl')):
        try:
            with open(f) as fp:
                for line in fp:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    strategy = r.get('strategy')
                    ts = r.get('timestamp')
                    if not strategy or not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts)
                        ts_ms = int(dt.timestamp() * 1000)
                    except Exception:
                        continue
                    prev = strategy_earliest_trade_ts.get(strategy)
                    if prev is None or ts_ms < prev:
                        strategy_earliest_trade_ts[strategy] = ts_ms
                    if r.get('trade_type') != 'exit':
                        continue
                    pnl = r.get('pnl_dollars')
                    if pnl is None:
                        continue
                    all_exits.append((ts_ms, strategy, float(pnl)))
        except Exception as e:
            print(f'WARN: failed reading {f}: {e}', file=sys.stderr)

    all_exits.sort(key=lambda x: x[0])
    print(f'Total exits: {len(all_exits)}')
    if all_exits:
        print(f'Earliest exit: {datetime.fromtimestamp(all_exits[0][0] / 1000)}')
        print(f'Latest exit:   {datetime.fromtimestamp(all_exits[-1][0] / 1000)}')

    cumulative = {}
    strategy_first_exit_seen = {}
    lines = []
    for ts_ms, strategy, pnl in all_exits:
        label_str = LABELS.get(strategy)
        if not label_str:
            continue
        if strategy not in strategy_first_exit_seen:
            strategy_first_exit_seen[strategy] = ts_ms
            baseline_ts = ts_ms - 3600 * 1000
            lines.append(f'{METRIC}{label_str} 0 {baseline_ts}')
            # Equity baseline: at the moment just before the first exit,
            # equity equals initial_capital (no realized, ~0 unrealized).
            if strategy in EQUITY_BACKFILL_STRATEGIES:
                initial = EQUITY_INITIAL_CAPITAL_USD.get(strategy, 0.0)
                lines.append(f'{EQUITY_METRIC}{label_str} {initial} {baseline_ts}')
        cumulative[strategy] = cumulative.get(strategy, 0.0) + pnl
        lines.append(f'{METRIC}{label_str} {cumulative[strategy]} {ts_ms}')
        # Equity at this exit timestamp = initial + cumulative realized
        # (omits other-symbol unrealized, see module docstring).
        if strategy in EQUITY_BACKFILL_STRATEGIES:
            initial = EQUITY_INITIAL_CAPITAL_USD.get(strategy, 0.0)
            equity = initial + cumulative[strategy]
            lines.append(f'{EQUITY_METRIC}{label_str} {equity} {ts_ms}')

    # For strategies with trade-log entries but no exits yet (e.g. CSCM
    # while the rebalance never fires), emit a $0 baseline anchored to
    # the earliest trade timestamp so the chart shows a continuous flat
    # zero line instead of a single live dot.
    for strategy, earliest_ts in strategy_earliest_trade_ts.items():
        if strategy in cumulative:
            continue
        label_str = LABELS.get(strategy)
        if not label_str:
            continue
        baseline_ts = earliest_ts - 3600 * 1000
        lines.append(f'{METRIC}{label_str} 0 {baseline_ts}')
        cumulative[strategy] = 0.0

    print(f'Generated {len(lines)} datapoints across {len(cumulative)} strategies')
    for s, v in sorted(cumulative.items()):
        equity_note = ''
        if s in EQUITY_BACKFILL_STRATEGIES:
            initial = EQUITY_INITIAL_CAPITAL_USD.get(s, 0.0)
            equity_note = f'  (equity backfilled = initial ${initial:,.0f} + lifetime ${v:.2f})'
        print(f'  {s}: ${v:.2f}{equity_note}')

    body = ('\n'.join(lines) + '\n').encode('utf-8')
    req = urllib.request.Request(
        VM_URL, data=body,
        headers={'Content-Type': 'text/plain'},
        method='POST',
    )
    with urllib.request.urlopen(req) as resp:
        print(f'VM response: {resp.status} {resp.reason}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
