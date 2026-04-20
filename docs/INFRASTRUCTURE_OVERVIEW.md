# Homeguard Trading Bot - Infrastructure Overview

**Last Updated**: 2026-04-20
**Region**: us-east-1 (N. Virginia)
**Instance**: t4g.medium ARM64 (4 GB RAM) running Amazon Linux 2023
**Estimated Monthly Cost**: ~$15-18 (see breakdown below)

> Placeholders like `<YOUR_EC2_IP>`, `<YOUR_INSTANCE_ID>` are resolved from `.env` or `terraform output`.

---

## Quick Summary

| Component | Status | Details |
|-----------|--------|---------|
| **EC2 Instance** | [+] Running | `<YOUR_INSTANCE_ID>` (t4g.medium, 4 GB RAM) |
| **Root Volume** | [+] Active | 50 GB gp3 (encrypted) |
| **Public IP** | [+] Active | `<YOUR_EC2_IP>` (Elastic IP) |
| **Remote Access** | [+] Tailscale | SSH and Grafana via Tailnet; public SSH restricted to `<YOUR_IP_CIDR>` |
| **Scheduled Start/Stop** | [+] Enabled | 9:00 AM - 4:30 PM ET Mon-Fri (equities). CSCM runs weekly Sun 00:00 UTC. |
| **Trading Services** | [+] Running | OMR, RAMP, CSCM under `homeguard-trading.target` |
| **Monitoring Stack** | [+] Running | VictoriaMetrics + Grafana + Loki + Promtail + node_exporter |

---

## Runtime Topology on EC2

All services run as systemd units on the single `t4g.medium` instance.

### Strategy services (trading)

| Unit | Strategy | Metrics Port | Schedule | Broker |
|------|----------|--------------|----------|--------|
| `homeguard-omr.service` | Overnight Mean Reversion | `8081` | Entry 3:50 PM, exit 9:31 AM ET | Alpaca paper |
| `homeguard-ramp.service` | Regime-Aware Momentum Protection | `8082` | Rebalance 3:55 PM ET | Alpaca paper |
| `homeguard-cscm.service` | Cross-Sectional Crypto Momentum | `8084` | Rebalance weekly Sun 00:00 UTC | DemoBroker (Binance WS prices, simulated fills) |
| `homeguard-trading.target` | Target aggregating the three above | - | - | - |

Legacy `homeguard-mp.service` (port 8083) exists but is superseded by RAMP; kept only as a fallback for ad-hoc reruns and is not started by `homeguard-trading.target`.

**Environment flags per service:**
- `METRICS_PORT` — per-strategy Prometheus scrape port
- `ENABLE_METRICS=true` — starts the `MetricsRegistry` HTTP thread
- `USE_STREAMING=true` (RAMP only) — WebSocket live bars from Alpaca
- `CSCM_USE_DEMO_BROKER=true` (CSCM only) — forces the DemoBroker path

### Monitoring services

| Unit | Role | Port | Data Path |
|------|------|------|-----------|
| `victoria-metrics.service` | TSDB (Prometheus-compatible) | `8428` | `/var/lib/victoria-metrics` |
| `grafana-server.service` | Dashboards + alerts | `3000` | `/var/lib/grafana` |
| `loki.service` | Log aggregator | `3100` | `/var/lib/loki` |
| `promtail.service` | Ships journald → Loki | - | - |
| `node-exporter.service` | Host CPU/mem/disk metrics | `9100` | - |
| `homeguard-snapshot.timer` | JSON metrics snapshot fallback | - | `/home/ec2-user/stock_data/metrics_snapshots/` |
| `homeguard-weekly-report.timer` | QuantStats weekly report | - | Sunday 00:30 UTC |
| `tailscaled.service` | Mesh VPN for remote access | - | - |

**Retention:**
- VictoriaMetrics: 90 days
- Loki: 14 days (configurable in `config/monitoring/loki/config.yaml`)
- Trade log JSONL: rolls daily under `/home/ec2-user/logs/trades_YYYYMMDD.jsonl`, retained indefinitely (small, append-only)

### Observability services (pre-existing)

| Unit | Role |
|------|------|
| `homeguard-discord.service` | Read-only Discord observability bot (Claude-powered) |
| `homeguard-gateway.service` | IBKR IB Gateway for options/equity orders via `ib_async` |
| `homeguard-xvfb.service` | Virtual framebuffer for the headless IB Gateway GUI |

---

## Infrastructure Diagram

```
                  Tailnet (Tailscale VPN)
                  --------------------
                  |  Operator laptop  |
                  --------------------
                           |
                  (tailnet IP, 100.x.y.z)
                           |
                           v
+--------------------------------------------------------------------+
|    EC2 t4g.medium (ARM64, 4 GB RAM) - Amazon Linux 2023            |
|    <YOUR_INSTANCE_ID>  /  <YOUR_EC2_IP>                            |
|                                                                    |
|   +------------------------+   +-------------------------------+   |
|   |  STRATEGY SERVICES     |   |  MONITORING STACK             |   |
|   |  (homeguard-trading)   |   |                               |   |
|   |  homeguard-omr   :8081 |-->|  victoria-metrics :8428       |   |
|   |  homeguard-ramp  :8082 |-->|   (scrapes every 15s)         |   |
|   |  homeguard-cscm  :8084 |-->|                               |   |
|   +------------------------+   |  grafana-server   :3000       |   |
|             |                  |  loki             :3100       |   |
|             v                  |  promtail  (journald shipper) |   |
|   +------------------------+   |  node-exporter    :9100       |   |
|   |  TRADE / STATE         |   +-------------------------------+   |
|   |  /home/ec2-user/logs/  |               |                       |
|   |    trades_YYYYMMDD.jsonl               v                       |
|   |  /home/ec2-user/stock_data/   Weekly QuantStats report         |
|   |    metrics_snapshots/         (timer: Sun 00:30 UTC)           |
|   +------------------------+                                       |
|                                                                    |
|   Root volume: 50 GB gp3 (encrypted, delete_on_termination=false)  |
+--------------------------------------------------------------------+
                           |
                           |  egress: Alpaca REST+WS, yfinance, IBKR Gateway,
                           |          Binance WS (CSCM), Anthropic API (Discord),
                           |          GitHub
                           v
```

Lambda + EventBridge control-plane (unchanged):

```
EventBridge cron                    EventBridge cron
 0 14 ? * MON-FRI *                  30 21 ? * MON-FRI *
   (9:00 AM ET)                        (4:30 PM ET)
       |                                  |
       v                                  v
  Lambda: homeguard-start-instance   Lambda: homeguard-stop-instance
       |                                  |
       +----------+          +------------+
                  v          v
              ec2:StartInstances / ec2:StopInstances
```

Lambda schedule only applies to equity-trading hours (OMR, RAMP). CSCM runs on a weekly cadence; the instance is already up during its Sunday tick because Lambda doesn't stop it on weekends (or we manually keep it up — see "CSCM note" below).

---

## Resource Breakdown

### Compute
1. **EC2 Instance** (`aws_instance.homeguard_trading`)
   - Type: **t4g.medium** (ARM64, 2 vCPU, 4 GB RAM)
   - AMI: Amazon Linux 2023 (ARM64)
   - Upgrade from t4g.small was needed after IB Gateway + monitoring stack pushed memory pressure above the 2 GB limit.

2. **EBS Volume**
   - **50 GB gp3**, encrypted, delete_on_termination=false
   - Was 8 GB; expanded to 50 GB on 2026-04-20 (online via `modify-volume` + `growpart` + `xfs_growfs`).
   - Sized for VM (90d × ~630 series ≈ 200 MB), Loki (~200 MB headroom), trade logs, IB Gateway working set, and ~40 GB free for growth.

3. **Elastic IP** — static IP persisted across restarts.

### Networking
4. **Security Group** `homeguard-trading-bot-sg`
   - Ingress: `22/tcp` from `<YOUR_IP_CIDR>` only. Grafana/VM/Loki are NOT exposed publicly — access routes through Tailscale.
   - Egress: all.

### Serverless Scheduling
5-6. **Start/Stop Lambdas** (Python 3.11) triggered by EventBridge cron — unchanged from original deployment.

### IAM / Logging / Alerts
- Lambda execution role with `ec2:StartInstances`, `ec2:StopInstances`, CloudWatch Logs.
- CloudWatch log groups for Lambda (90d retention).
- Optional CloudWatch Agent for host metrics is **disabled**; node_exporter + VM replaces it.
- Optional SNS topic for Lambda failure alerts (off by default).

---

## Remote Access

Primary path: **Tailscale**. The EC2 host and the operator laptop join the same tailnet. Grafana and SSH are bound to the tailnet address only — no public ingress required beyond the `<YOUR_IP_CIDR>` SSH fallback.

Grafana URL (tailnet-only): `http://<ec2-tailnet-ip>:3000`
VictoriaMetrics UI: `http://<ec2-tailnet-ip>:8428/vmui`

Public SSH fallback still works via Elastic IP:
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>
```

---

## Cost Estimate

| Component | Notes | Monthly |
|-----------|-------|---------|
| EC2 t4g.medium | ~157 hrs/month at $0.0336/hr (Lambda-scheduled) | $5.29 |
| EBS 50 GB gp3 | $0.08/GB | $4.00 |
| Elastic IP | charged when instance is stopped | $3.60 |
| Lambda invocations | free tier | ~$0.01 |
| CloudWatch Logs (Lambda only) | ~10 MB | ~$0.01 |
| Data transfer | ~100 MB out | $0.10 |
| Tailscale | free tier (personal plan) | $0.00 |
| **Total** | | **~$13/mo** |

*Adjust upward if the instance starts running 24/7 (e.g. for CSCM's Sunday tick) — t4g.medium 24/7 ≈ $24/mo on compute alone.*

**CSCM note:** the Lambda stop schedule only fires Mon-Fri 4:30 PM ET, so the instance is still up on Saturdays and Sundays to catch CSCM's Sunday 00:00 UTC rebalance. If you adopt a tighter schedule, ensure the instance is started before that tick or move CSCM to its own schedule.

---

## Daily Operation Flow

### Monday-Friday

- **9:00 AM ET** — EventBridge → start Lambda → EC2 boot (~30s). systemd brings up `homeguard-trading.target`, monitoring stack, IB Gateway.
- **9:30 AM ET** — equity market opens. OMR exits overnight positions shortly after open.
- **3:50 PM ET** — OMR enters overnight positions.
- **3:55 PM ET** — RAMP rebalances via IBKR historical bars + Alpaca trading.
- **4:00 PM ET** — market closes. Monitoring stack continues collecting.
- **4:30 PM ET** — EventBridge → stop Lambda → graceful shutdown. EBS + Elastic IP persist.

### Weekends (Sat-Sun)
- Instance remains up (not scheduled to stop) so CSCM can fire at Sun 00:00 UTC.
- CSCM rebalance runs: pulls crypto prices from DemoBroker (Binance WS), applies cross-sectional momentum + BTC regime filter, executes simulated fills through DemoBroker's simulated slippage + fees.
- `homeguard-weekly-report.timer` fires Sun 00:30 UTC: generates QuantStats HTML report from the trade log JSONL and posts a summary.

---

## Management

### SSH / ops scripts
Location: `infra/ec2/` (Windows `.bat` + Unix `.sh`):

| Script | Purpose |
|--------|---------|
| `connect.{bat,sh}` | SSH into instance |
| `check_bot.{bat,sh}` | Check all homeguard-* service statuses |
| `view_logs.{bat,sh}` | Stream `journalctl` for the trading target |
| `restart_bot.{bat,sh}` | Restart `homeguard-trading.target` |
| `daily_health_check.{bat,sh}` | 6-point health check |
| `local_start_instance.bat` / `local_stop_instance.bat` | Manual EC2 start/stop |

SSH aliases configured on `.ssh/config`: `bot-status`, `bot-logs`, `bot-logs-recent`, `bot-update`, `bot-restart`.

### systemd commands

```bash
# Whole trading stack
sudo systemctl status homeguard-trading.target
sudo systemctl restart homeguard-trading.target

# Individual strategy
sudo systemctl restart homeguard-omr
sudo journalctl -u homeguard-ramp -f

# Monitoring stack
sudo systemctl status victoria-metrics grafana-server loki promtail node-exporter
```

### Code updates
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> \
  "cd ~/Homeguard && git pull && sudo systemctl restart homeguard-trading.target"
```

Dashboard JSON changes under `config/monitoring/grafana/dashboards/` require copying to `/var/lib/grafana/dashboards/homeguard/` and reloading `grafana-server`.

### Log locations on EC2

| What | Where |
|------|-------|
| Strategy stdout | `journalctl -u homeguard-{omr,ramp,cscm}` → Loki via Promtail |
| Trade events (JSONL) | `/home/ec2-user/logs/trades_YYYYMMDD.jsonl` |
| Metrics snapshots (JSON) | `/home/ec2-user/stock_data/metrics_snapshots/{strategy}_snapshot.json` |
| VictoriaMetrics data | `/var/lib/victoria-metrics/` |
| Loki data | `/var/lib/loki/` |
| Grafana data + dashboards | `/var/lib/grafana/` |
| Lambda start/stop logs | CloudWatch: `/aws/lambda/homeguard-{start,stop}-instance` |

**Note:** `data/trading/logs/snapshots/portfolio_history.csv` is legacy (last write 2026-03-25). `PortfolioLogger`/`PortfolioSnapshotWorker` are no longer wired into the runners; the metrics snapshot + VictoriaMetrics replaces them. Either re-wire or delete those modules in a future cleanup.

---

## Security Summary

- **Network**: SSH restricted to `<YOUR_IP_CIDR>`. Grafana/VM/Loki bound to tailnet (100.x.y.z) — no public ingress on 3000/8428/3100.
- **Data at rest**: EBS encrypted, IAM least-privilege on Lambda role, IMDSv2 required.
- **Secrets**: `.env` on instance (Alpaca, Discord, Anthropic keys, IBKR config). Not committed.
- **Tailscale**: separate auth plane; compromised Tailnet key is scoped only to the tailnet devices and revocable from the admin console.

---

## Terraform state

| File | Role |
|------|------|
| `infra/terraform/main.tf` | Instance, security group, Elastic IP, optional SNS/CloudWatch |
| `infra/terraform/variables.tf` | `instance_type=t4g.medium`, `root_volume_size=50` (both defaults) |
| `infra/terraform/scheduled_start_stop.tf` | Lambda + EventBridge cron |
| `infra/terraform/monitoring.tf` | Optional CloudWatch Agent (off by default) |
| `infra/terraform/outputs.tf` | EC2 ID, public IP, DNS |

The monitoring stack (VM, Grafana, Loki, Promtail, node_exporter, Tailscale) is **not** managed by Terraform — it's installed idempotently via `infra/ec2/setup/install_*.sh`, with service units in `infra/ec2/services/*.service`. That's intentional: these are OS-level daemons, not AWS resources, and bootstrapping them via Terraform would obscure the config files under `config/monitoring/`.

---

**Managed by**: Terraform (AWS resources) + setup scripts under `infra/ec2/setup/` (instance services)
**Monitoring design spec**: `docs/superpowers/specs/2026-04-18-monitoring-system-design.md`
**Metric naming contract**: `docs/monitoring/METRIC_SPEC.md`
