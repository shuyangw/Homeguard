#!/usr/bin/env python3
"""Backfill hg_strategy_realized_pnl_lifetime_usd into VictoriaMetrics from trade logs.

Reads all trades_*.jsonl in the log dir chronologically, computes per-strategy
cumulative realized PnL at each exit timestamp, and POSTs as a series of
timestamped Prometheus-format gauge values to VM's /api/v1/import/prometheus
endpoint.

Idempotent: re-running with the same input emits the same (metric, labels,
timestamp, value) tuples. VM dedupes on (series, timestamp) so re-imports are
safe.

When to run:
- After deploying the lifetime-PnL feature for the first time, to backfill
  trade-log history that pre-dates the gauge's live emission. The dashboard
  panel `Cumulative Realized P&L by Strategy` reads this gauge directly, so
  without backfill it would only show data going forward from deploy.
- After a VM data-loss event, to restore historical points (limited by VM's
  retention period, default 90d in our setup).

Note: VM silently drops imports older than its retention window. Today
that's ~90d, so trades older than ~3 months won't appear in queries.

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

LABELS = {
    'ramp': '{instance="127.0.0.1:8082",job="homeguard-ramp",strategy="ramp"}',
    'cscm': '{instance="127.0.0.1:8084",job="homeguard-cscm",strategy="cscm"}',
    'omr':  '{instance="127.0.0.1:8081",job="homeguard-omr",strategy="omr"}',
    'mp':   '{instance="127.0.0.1:8083",job="homeguard-mp",strategy="mp"}',
}


def main() -> int:
    all_exits = []
    for f in sorted(LOG_DIR.glob('trades_*.jsonl')):
        try:
            with open(f) as fp:
                for line in fp:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get('trade_type') != 'exit':
                        continue
                    pnl = r.get('pnl_dollars')
                    strategy = r.get('strategy')
                    ts = r.get('timestamp')
                    if pnl is None or not strategy or not ts:
                        continue
                    dt = datetime.fromisoformat(ts)
                    ts_ms = int(dt.timestamp() * 1000)
                    all_exits.append((ts_ms, strategy, float(pnl)))
        except Exception as e:
            print(f'WARN: failed reading {f}: {e}', file=sys.stderr)

    all_exits.sort(key=lambda x: x[0])
    if not all_exits:
        print('No exits to backfill')
        return 0

    print(f'Total exits: {len(all_exits)}')
    print(f'Earliest:    {datetime.fromtimestamp(all_exits[0][0] / 1000)}')
    print(f'Latest:      {datetime.fromtimestamp(all_exits[-1][0] / 1000)}')

    cumulative = {}
    strategy_first_seen = {}
    lines = []
    for ts_ms, strategy, pnl in all_exits:
        label_str = LABELS.get(strategy)
        if not label_str:
            continue
        if strategy not in strategy_first_seen:
            strategy_first_seen[strategy] = ts_ms
            # $0 baseline 1 hour before first exit so the chart starts at zero.
            baseline_ts = ts_ms - 3600 * 1000
            lines.append(f'{METRIC}{label_str} 0 {baseline_ts}')
        cumulative[strategy] = cumulative.get(strategy, 0.0) + pnl
        lines.append(f'{METRIC}{label_str} {cumulative[strategy]} {ts_ms}')

    print(f'Generated {len(lines)} datapoints across {len(cumulative)} strategies')
    for s, v in sorted(cumulative.items()):
        print(f'  {s}: ${v:.2f}')

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
