# Runbook: Backfill Grafana Gaps After a Trading-Instance Pause

When the trading EC2 has been stopped for a period (no scrapes, no metric emission), the `portfolio_overview` dashboard shows gaps on the `Equity Curve`, `Drawdown %`, `Day P&L`, and `RAMP Regime State` panels. This runbook restores visual continuity.

**Time required:** ~10 minutes after EC2 is back online.

---

## 1. Backfill the regime-state series

`hg_regime_state_code` and its companions can be safely recomputed from SPY+VIX history. The script is idempotent (re-runs are no-ops on overlapping windows).

```bash
ssh ec2-user@<trading-ec2>
cd ~/Homeguard
source venv/bin/activate
python -m scripts.ops.backfill_regime_state
```

Note: Use the `-m` module form (not `python scripts/ops/backfill_regime_state.py`) so the project root is on `sys.path` for the `src.*` imports.

Optional flags:
- `--since YYYY-MM-DD` — start of the window. Defaults to the earliest existing sample in VM for `hg_regime_state_code{job="homeguard-ramp"}`.
- `--until YYYY-MM-DD` — end of the window. Defaults to today (UTC).
- `--dry-run` — print the OpenMetrics body to stdout without writing to VM. Use this first to eyeball output before the real run.

**Verify in Grafana:** open `portfolio_overview` -> `RAMP Regime State Code` panel. The series should now be continuous across the previously-gapped window.

---

## 2. Annotate the equity/drawdown/P&L gaps

These series cannot be safely backfilled (see `scripts/ops/backfill_lifetime_pnl.py:24-31` for the three-formula-era history). Instead, mark each gap with a Grafana annotation.

### Step 2a. Append the gap window to the source of truth

Edit `config/monitoring/grafana/annotations/instance_off.json` and append a new entry:

```json
{
  "start": "YYYY-MM-DDTHH:MM:SSZ",
  "end":   "YYYY-MM-DDTHH:MM:SSZ",
  "text":  "Trading instance offline"
}
```

Use UTC timestamps. `start` = approximate time the EC2 stopped scraping; `end` = approximate time scraping resumed. Approximate is fine; the annotation overlay is for context, not audit.

### Step 2b. Commit the change

```bash
git add config/monitoring/grafana/annotations/instance_off.json
git commit -m "ops(monitoring): record instance-off window YYYY-MM-DD..YYYY-MM-DD"
```

### Step 2c. Sync into Grafana

Run from EC2 (Grafana is on `127.0.0.1:3000` there):

```bash
python -m scripts.ops.sync_grafana_annotations \
  --source config/monitoring/grafana/annotations/instance_off.json \
  --grafana-url http://127.0.0.1:3000 \
  --api-key "$GRAFANA_API_KEY" \
  --dashboard-uid portfolio-overview
```

`GRAFANA_API_KEY` should be exported by `infra/ec2/load_env.sh`. If it's missing, generate one via `Grafana -> Administration -> Service accounts`.

The sync is idempotent: re-running with no spec changes produces `0 new, N already present`.

**Verify in Grafana:** Equity / Drawdown / Day P&L panels show a shaded red region with hover-tip text over the documented window.

---

## 3. What this runbook does NOT do

- It does not rewrite `hg_strategy_equity_usd` or `hg_portfolio_day_pnl_usd` historical samples. Those have a three-formula-era history (see `scripts/ops/backfill_lifetime_pnl.py:24-31`); the annotation overlay is the chosen approach.
- It does not prevent future gaps. If continuous monitoring during pauses is required, see the "Out-of-scope" section in `docs/superpowers/specs/2026-05-16-grafana-gap-backfill-design.md` for the "detached metrics stack" option.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `backfill_regime_state` aborts with `VIX data unavailable (^VIX)` | Both Alpaca and yfinance failed for `^VIX` | Retry. If persistent, check Alpaca subscription status and yfinance availability. |
| Script aborts with `No --since provided and VM has no existing samples` | First-ever backfill, nothing in VM yet | Re-run with explicit `--since YYYY-MM-DD` corresponding to when RAMP first emitted regime data. |
| Script aborts with `VM unreachable at http://127.0.0.1:8428/...` | VictoriaMetrics is not running on the EC2 instance | Start the `victoria-metrics` systemd service before re-running. |
| `sync_grafana_annotations` exits 0 but panels still ungapped | Grafana caching, or wrong `--dashboard-uid` | Hard-refresh the dashboard. Verify the dashboard UID matches `portfolio-overview` in the Grafana UI (Dashboard settings -> JSON Model). |
| Regime panel shows continuous series but flat-lined at value 0 | Detector returned `'SIDEWAYS'` fallback or unknown regime mapping fired (state_code=0) | Confirm SPY/VIX fetch returned >= 252 days of data prior to the start of the backfill window. Check the script logs for `[RAMP] Insufficient ... data` warnings. |

---

## Related

- Spec: `docs/superpowers/specs/2026-05-16-grafana-gap-backfill-design.md`
- Plan: `docs/superpowers/plans/2026-05-16-grafana-gap-backfill.md`
- Live emission reference: `scripts/trading/run_live_paper_trading.py:725-774` (`_emit_strategy_specific_metrics`)
- VM import endpoint reference: `scripts/ops/backfill_lifetime_pnl.py`
