# Monitoring + Infra Updates - 2026-04-20

## Summary
Fixed two live-trading bugs (IBKR daily bar tz coercion, CSCM broker switch to DemoBroker), repaired PnL/drawdown dashboards that had been masking negative values, expanded the EC2 root volume from 8 GB to 50 GB online, and refreshed architecture documentation to reflect the deployed monitoring stack.

## Changes Made

- **`src/trading/brokers/ibkr/data_download.py`** (commit `6ca32b8`): ib_async returns `date` objects for daily bars (plain `Index`, no `.tz`) and tz-aware datetimes for intraday. Added a `pd.to_datetime()` coercion before timezone arithmetic so `IBKRDataProvider._normalize_dataframe` doesn't crash on daily bars.
- **CSCM live adapter** (commit `2115e7f`): switched to `DemoBroker` (Binance WebSocket streaming with simulated fills and fractional qty) for paper trading. Replaces the prior CryptoBrokerRouter in the CSCM runtime path.
- **`config/monitoring/grafana/dashboards/portfolio_overview.json` + `incident_review.json`** (commit `b2e8fed`): the drawdown and PnL panels wrapped the metric in `max(...)`, which clamps per-strategy negatives. Switched both to raw `hg_portfolio_drawdown_pct` with `legendFormat: "{{job}}"` and `tooltip.mode: multi`. Day/realized PnL panels already correctly sum signed values; left those alone. Now losses show as negative bars in Grafana.
- **EC2 root volume**: online resize from 8 GB -> 50 GB gp3 via `aws ec2 modify-volume` + `sudo growpart /dev/nvme0n1 1` + `sudo xfs_growfs /`. No downtime. Result: 50 GB total / 43 GB free / 16% used. Unlocks headroom for 1Hz portfolio state scrape if we want it later.
- **`infra/terraform/variables.tf`**: bumped `root_volume_size` default from 8 -> 50 so the next `terraform apply` doesn't try to shrink the disk. Added a sizing note citing the monitoring stack retention.
- **`docs/INFRASTRUCTURE_OVERVIEW.md`**: full rewrite. Previous version documented t4g.small, 8 GB, a single `homeguard-trading.service`, and no monitoring stack - all wrong. New version covers three strategy services (OMR/RAMP/CSCM with metrics ports), six monitoring services (VM/Grafana/Loki/Promtail/node-exporter/Tailscale), retention policies (VM 90d, Loki 14d, trade JSONL indefinite), Tailscale-gated remote access, and the ~$13-18/mo cost breakdown.
- **`docs/architecture/ARCHITECTURE_OVERVIEW.md`**: updated the "Recently Deployed" section. Replaced the stale `homeguard-mp.service` reference with `homeguard-ramp.service`, added CSCM with DemoBroker, added a new bullet for the self-hosted monitoring stack pointing at `METRIC_SPEC.md` and the monitoring design spec.

## Commits
- `6ca32b8` fix(ibkr): coerce daily-bar date objects to DatetimeIndex
- `2115e7f` feat(cscm): switch to DemoBroker for paper trading
- `b2e8fed` fix(monitoring): break out portfolio PnL/drawdown per-strategy so charts go negative

## Known Issues / Remaining Work
- **Alpaca WebSocket 429**: intermittent rate-limit rejection on streaming subscribe. Currently non-fatal (falls back to REST polling). Worth a dedicated session to look at the retry/backoff logic in `src/streaming/`.
- **PortfolioLogger / PortfolioSnapshotWorker**: still in the codebase but no longer wired into the runtime path (last CSV write 2026-03-25). Superseded by VictoriaMetrics. Candidate for removal in a cleanup PR.
- **1Hz portfolio scrape**: disk now has room (~43 GB free), but I did not enable it. Current 15s scrape_interval is sufficient for the incident-review use case.

## Validation
- **Dashboards**: verified post-deploy in Grafana - Portfolio Overview "Drawdown % by Strategy" and "Day P&L" now render signed values; Incident Review panel shows per-strategy legend with negative drawdowns during the 2026-04-18 OMR drawdown window.
- **EBS expansion**: `df -hT /` post-resize shows `/dev/nvme0n1p1 xfs 50G 8.0G 43G 16%`. All three trading services and the monitoring stack continued running during the resize (zero restarts).
- **Terraform drift check**: `terraform plan` against the current default no longer proposes a volume shrink.
- **IBKR fix**: tested via `get_historical_bars(SPY, 1D)` locally - returns tz-aware ET DatetimeIndex as expected.
