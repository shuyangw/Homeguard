# Grafana Alerting Fix - 2026-07-27

## Summary

`config/monitoring/grafana/alerts/rules.yaml` defined 7 alert rules for a live
trading system and **none had ever been active** since the file was created on
2026-04-18. Found by exercising the Grafana MCP integration installed earlier the
same session: `alerting_manage_rules` returned `null`, which turned into an audit.
Six independent defects were found and fixed, rules are now provisioned and
evaluating, and delivery (contact points/routing) is deliberately deferred.

## The six defects

1. **Nothing loaded the file.** `install_grafana.sh` provisioned datasources and
   dashboards only. No `vmalert`, no Alertmanager. The file header said "Import
   via Grafana UI" from its first commit; `91a7297` (2026-06-09) said outright
   "Needs provisioning into Grafana". It never happened.
2. **Wrong schema.** Prometheus/vmalert format, not Grafana unified alerting.
3. **Drawdown sign inverted.** `hg_portfolio_drawdown_pct` is negative by
   construction, so `max(...) > 7` / `> 9` could never fire.
4. **Order metrics dead.** `hg_orders_rejected_total` was defined and called, but
   `StrategyAdapter` built `ExecutionEngine(broker)` with no registry.
5. **`max(hg_market_open) == 1` was a no-op.** Unlabeled gauge, and
   `run_cscm_live.py:229` hardcodes it to 1.0 for crypto, so `max()` across jobs
   is permanently 1. Three rules gated on it and would have fired every night
   and weekend. Worse than silence: alert spam trains you to ignore alerts.
6. **`StrategySignalStale` read the wrong metric.**
   `hg_strategy_last_signal_timestamp` is `time.time()` every tick, i.e. process
   liveness, not signal age.

## Key decisions

- **No drawdown alerting at all**, RAMP included. Per the operator: gradual
  losses are a dashboard concern. Reinforced by a hard finding: **no automated
  drawdown stop exists in the live path**, so the old annotation "Check if
  auto-stop triggered" referenced a mechanism that was never built. No discrete
  event to alert on, no action a page would prompt. Defect 3 closed by deletion.
- **Grafana file provisioning over vmalert+Alertmanager.** Two more daemons on a
  4GB box already capped at grafana 400M / VM 200M / tailscaled 100M is not worth
  it for 5 rules, and `noDataState: OK` cleanly expresses "empty vector means the
  gate is closed".
- **Not generalised across `homeguard-*` jobs.** `omr` and `mp` measure `up=0`
  (units disabled, superseded by `homeguard-multi`), so a regex rule would fire
  permanently. Service-down rules enumerate ramp and cscm only.
- **OMR order metrics deliberately not wired.** `OMRLiveAdapter` has no
  `metrics_registry` param and is `up=0`. Adding a parameter nothing passes would
  be aspirational. The rule annotation states the real coverage rather than
  claiming OMR is covered.

## Thresholds, set from measured data (not guessed)

| Measurement | Value | Consequence |
|---|---|---|
| ibkr heartbeat age, market hours, 7d | p99 71s, max 166s | 300s threshold (old 120s was inside the noise band) |
| ibkr heartbeat age, ungated, 2d | max 310s | the market-hours gate is load-bearing |
| `up` by job | ramp=1 cscm=1 node=1, omr=0 mp=0 | do not generalise service-down rules |
| RAMP decision age, 7d | ~24h weekdays, 72h over a weekend | bare age threshold impossible; needs a post-close time gate |
| CSCM decision age, 7d | climbs to 168h then resets | no useful staleness rule; liveness only |

## Changes Made

- **`config/monitoring/grafana/alerting/homeguard_rules.yaml`** (new): 7 rules in
  Grafana unified-alerting schema. `IBGatewayDown`, `WebSocketDisconnected`,
  `RampDecisionMissedToday`, `Ramp`/`CscmServiceDownOrCrashLooping`,
  `OrderRejectionSpike`, and an always-firing canary.
- **`config/monitoring/grafana/alerts/rules.yaml`**: deleted. Two sources of
  truth is how this stayed broken.
- **`infra/ec2/sync_grafana_alerts.sh`** (new): mirrors the dashboards sync but
  **restarts grafana-server**, because `provisioning/alerting/` has no file
  watcher. Guarded on a content compare, and greps the journal afterwards.
- **`infra/ec2/setup/install_grafana.sh`**, **`instance_update_repo.sh`**: hooks.
- **`src/trading/core/execution_engine.py`**: added `classify_reject_reason()`, a
  closed set of 8 classes plus `other`, replacing `str(e)[:50]`.
- **`src/trading/adapters/strategy_adapter.py`**: `metrics_registry` kwarg,
  forwarded to `ExecutionEngine`. `ramp_live_adapter.py` passes it via `super()`.
- **`scripts/monitoring/validate_alert_exprs.py`** (new): runs every expression
  against the datasource.
- **`docs/monitoring/METRIC_SPEC.md`**: documented the negative sign convention.
- Tests: `test_alert_rules_provisioning.py` (new, 9 guards),
  `test_order_metrics_wiring.py` (new), plus `test_registry.py` and
  `test_strategy_adapter_base.py` updates.

## Commits

On `main` (`b5284e9..6abaa71`):
- `8073acb` fix(trading): wire the metrics registry into ExecutionEngine
- `e35ca81` feat(monitoring): provision Grafana alert rules, fixing 6 defects
- `6abaa71` docs(monitoring): document the drawdown sign convention

On deploy branch `ramp-phase4-turnover-regime-research` (`0eafc19..b3a8316`):
- the three above, cherry-picked clean
- `b3a8316` fix(infra): bring install_grafana.sh in line with main

## Validation

**Every expression** run against live VictoriaMetrics: 7 ok, 0 failed, 4 empty.
Each empty proven to be a closed gate, not a typo, by decomposing it:
`hg_market_open{job="homeguard-ramp"}` = 0 (market closed), `hour()` = 2 UTC,
decision age 24990s < 43200.

**Defect 5 demonstrated side by side** on live data with the market closed:
`max(hg_market_open)` = **1** (old, broken) vs
`max(hg_market_open{job="homeguard-ramp"})` = **0** (fixed).

**Threshold semantics checked, not assumed.** `up == 0` preserves the left-hand
value and yields 0, so a `gt 0` threshold would never fire. Rules use
`== bool 0` so the value is explicitly 1 = down. This would have been a seventh
defect had it shipped.

**Schema round-tripped**: provisioned, then read back via
`/api/v1/provisioning/alert-rules`. `provenance=file`, and `condition`/`for`/
`noDataState`/`execErrState` plus the full A->B->C chain preserved exactly.

**A real rule proven to fire**: a temporary probe on
`min(hg_portfolio_drawdown_pct{job="homeguard-cscm"}) < -9` went to `firing` and
rendered `PROBE: CSCM drawdown -13.74% is below -9`. Proves the
query->reduce->threshold chain on real data, the negative-sign analysis, and that
`{{ $values.B.Value }}` templating works where the old `{{ $value }}` would have
rendered empty.

**Both linters mutation-tested.** Every guard was verified to fail when its
defect is reintroduced: unscoped market_open gate, positive drawdown threshold,
phantom metric, bad job label, duplicate uid, missing severity, `for:` as int,
missing nested `model.datasource`, and condition naming a query node instead of
the threshold node (that last guard was **missing** and was added after the
mutation test caught it). Reverting the `ExecutionEngine` wiring or the
classifier also fails the suite.

**Sync script**: ran on the host from the committed path. First run updated and
restarted; second run correctly skipped the restart.

**Final state**: 7 rules, 0 unhealthy, 6 inactive, canary firing.

**Test suite**: 1005 passed. The only 2 failures
(`test_broker_routing.py::test_load_returns_dict`,
`::test_default_broker_for_unlisted_strategy`) were verified **pre-existing** by
stashing all changes and re-running against `origin/main`.

## Delivery (Layer 1), added later the same session

Discord contact point plus a notification policy tree, provisioned through the
same file mechanism and sync script as the rules. Routing by severity: critical
notifies in 10s and repeats hourly; warning waits 5m and repeats every 12h; info
is the canary, routed with a 24h `repeat_interval` so it becomes a once-a-day
"alerting is still alive" heartbeat instead of a storm.

The webhook URL is not committed. Grafana interpolates `${VAR}` in provisioning
files from its process environment, supplied by a new
`EnvironmentFile=-/etc/homeguard/grafana.env` in `grafana-server.service`.
Interpolation was verified specifically for alerting provisioning by probing a
non-secret field, because Grafana redacts the `url` field in API responses.

### INCIDENT: I took Grafana down

Provisioning the contact point before the secret existed **crash-looped
grafana-server for several minutes**.

An unset `${VAR}` does not degrade to "provisioned but undeliverable", which is
what I had assumed and documented. Grafana rejects a discord integration with an
empty url, its provisioning module fails, and the process exits:

```
Failed to provision alerting: failure parsing contact points: homeguard-discord:
could not find webhook url property in settings
grafana-server.service: Main process exited, code=exited, status=1/FAILURE
```

Dashboards and rule evaluation went down with it. **Trading was unaffected**
(`homeguard-multi`, `homeguard-cscm`, `victoria-metrics`, `loki` all stayed
active). Service was restored by moving the file aside and restarting; verified
back to 7 rules, 0 unhealthy, canary firing.

My missing-secret check had *warned* and then installed the file anyway. It now
**refuses** to install any provisioning file with an unresolved variable, and
removes a previously-installed copy so a later restart or reboot cannot fail for
a reason nobody connects to the change. Verified on the host: the file is skipped,
Grafana stays active, only `homeguard_rules.yaml` is installed.

The transferable lesson: for config a service reads **at startup**, an invalid
value is not a degraded feature, it is a boot failure. Fail closed on the
feature, not open on the service.

### Delivery CONFIRMED working

End to end verified: Grafana -> Discord -> `#homeguard-status`. Proven from both
sides, not just by absence of an error.

- `POST /api/alertmanager/grafana/config/api/v1/receivers/test` -> HTTP 200
- Read back via the bot: `GET /channels/1531481095441481831/messages` returned
  the message, author `Grafana`, title `[FIRING:1] HomeguardDeliveryTest`
- Operator confirmed it visually

Final state: Grafana healthy, 7 rules, 0 unhealthy, canary firing, contact points
= `homeguard-discord` (discord) plus Grafana's built-in email receiver.

Getting there surfaced three things worth recording:

1. **A 404 "Unknown Webhook" is not always a bad URL.** The first installed
   webhook 404'd because it had been deleted while being reconfigured, minutes
   after I had successfully validated it. Re-validating showed it live again and
   pointing at the intended channel.
2. **Changing a webhook's channel in the Discord UI preserves the URL.** So
   retargeting needs no config change on this side. Note the reverse is not true:
   `channel_id` cannot be changed through the token-authenticated API route, only
   via bot auth with Manage Webhooks.
3. **"No error in the log" is not proof of delivery.** After the credential was
   fixed, the 404 stopped appearing and it looked like delivery had started. It
   had not: reading the channel showed zero messages. The canary's notification
   was suppressed because alertmanager had already recorded an attempt for that
   aggregation group, and the info route's `repeat_interval: 24h` means the next
   attempt is 24h out. Positive confirmation required reading the destination.

That last point is a real weakness of the 24h heartbeat interval: a delivery that
fails once waits a day for the next attempt. Acceptable for a heartbeat whose
purpose is "is the path alive", but it means the heartbeat cannot be used to
diagnose a freshly-broken path quickly. Use the test endpoint for that.

Also cleaned up: a `zz-interp-test` contact point left in Grafana's database by
the interpolation experiment. Deleting its provisioning file did not remove it
(same behaviour as rules), so it needed
`DELETE /api/v1/provisioning/contact-points/{uid}` with `X-Disable-Provenance`.

### Credential hygiene follow-up

Two webhook URLs passed through the setup conversation, so both should be treated
as exposed. The unused one (`#general`, "Captain Hook") should be deleted in
Discord if it still exists. Rotating the live one is a one-line change to
`/etc/homeguard/grafana.env` plus a `sync_grafana_alerts.sh` run, with no code
change and no redeploy.

### Previously blocked on one manual step (now resolved)

The Discord bot (`Homeguard-Bot`, guild `Homeguard`) authenticates fine, but
returns `403 Missing Permissions` on `/channels/{id}/webhooks`, so the webhook
could not be created programmatically. Also note there is **no `#alerting`
channel**; the guild has `#general`, `#homeguard-querying`, and
`#homeguard-status`. Target chosen: `#homeguard-status`
(id `1531481095441481831`).

To finish, either grant the bot Manage Webhooks on that channel (keeps the
credential out of any transcript), or create the webhook by hand and write it to
`/etc/homeguard/grafana.env`, then re-run `sync_grafana_alerts.sh`.

Unrelated bug spotted in passing: `/etc/systemd/system/homeguard-discord.service:21`
uses `StartLimitIntervalSec` in the `[Service]` section, where systemd ignores it
(`Unknown key name ... ignoring`). It belongs in `[Unit]`.

## Known Issues / Remaining Work

- **Nothing is delivered anywhere.** No contact points, no notification policy.
  Rules evaluate and show state; no page happens. `severity` labels are inert.
  This is the deferred next conversation.

  **Correction to an earlier claim in this document:** it originally said
  `DISCORD_MONITORING_WEBHOOK` "already exists on the host and is used by
  `weekly_report.py`, so a Grafana Discord contact point can reuse it". That is
  wrong. The host `.env` contains only `DISCORD_TOKEN` (the bot token, a
  different thing). `DISCORD_MONITORING_WEBHOOK` is unset, so
  `weekly_report.py` has been taking its "webhook not configured" branch and the
  weekly QuantStats report has never been delivered either. That is the same
  silent-delivery failure class as the alerting bug, in a second place. There was
  no existing channel to reuse; delivery had to be built.
- **Host-level death is structurally uncoverable** by in-process Grafana
  alerting: if the box dies the notifier dies with it, and Grafana only runs
  08:00-20:00 ET weekdays plus a Sat 23:00-Sun 00:10 UTC window. Needs an
  off-box watchdog.
- **`homeguard-multi` restarted 2026-07-28 00:26 EDT** (15.5h before the 15:55 ET
  rebalance, market closed). Clean: `active`, `NRestarts: 0`, no crash-loop. The
  `code=10167` IBKR lines in the startup log are "market data not subscribed /
  delayed data" notices, normal for a paper account outside market hours.

  State verified non-destructive. Position and equity gauges came back
  **byte-identical** to the pre-restart capture (17 position series), and
  `hg_strategy_last_decision_timestamp` is unchanged at `1785182149.2468166`.
  Note the gauges read **empty for the first ~25s** after restart, because the
  registry is in-memory and repopulates on the metrics tick; that is not state
  loss, and the positions live at IBKR regardless. Do not conclude anything from
  a metrics read taken immediately after a restart.

  Unplanned validation: the restart took the RAMP metrics target down for roughly
  a minute, and `RampServiceDownOrCrashLooping` correctly stayed `inactive`. Its
  `for: 5m` absorbed a clean restart, which is exactly what that tolerance is
  documented to be for. No spurious notification was sent.

  **Still unverified:** `hg_orders_*` remains absent. The registry only exports a
  counter after its first increment, and RAMP places orders once daily at
  15:55 ET, so `OrderRejectionSpike` cannot be confirmed live until after today's
  rebalance. Check then that `hg_orders_submitted_total` and
  `hg_orders_filled_total` exist with a `reason` label drawn only from the closed
  classifier set.
- **Tests write to `config/trading/strategy_toggle.yaml`.** Running
  `tests/trading/` mutates it via `StrategyStateManager` with the default
  `modified_by='api'`, bumping `last_modified`. Enabled values are preserved, but
  this dirties a gitignored + force-tracked live runtime config on every test
  run. Reverted manually here; it is a test-isolation bug worth fixing.
- **`main` has diverged locally.** A concurrent session holds 4 unpushed
  options-docs commits on local `main` (tip `146ba34`) while `origin/main` is at
  `6abaa71`. Nothing was lost, my push was a clean fast-forward from `b5284e9`,
  but that session needs to rebase before pushing, and a force-push there would
  drop these three commits.
- **`sample.yaml`** (Grafana's own shipped file) sits in
  `/etc/grafana/provisioning/alerting/` on the host, dated from install. Inert
  (rule count is 7, all ours), but worth knowing it is there.
- **Phantom risk limits, flagged and NOT actioned.**
  `config/trading/omr_trading_config.yaml:76-85` declares `max_daily_loss_pct`,
  `max_weekly_loss_pct` and `max_drawdown_pct` with comments saying "triggers
  halt". They load into attributes (`omr_config_loader.py:91-94`) with **zero
  readers**, and `position_manager.check_stop_losses()` is dead code. Same block
  in `omr_expanded_config.yaml` and `momentum_trading_config.yaml`. The repo
  asserts risk protection it does not have. Needs a decision: delete the dead
  config, implement a real auto-stop that writes `set_enabled(False)` and emits
  an `hg_strategy_halted` gauge, or document as aspirational.
- **`RampDecisionMissedToday`'s `hour() >= 22` UTC window** is coupled to the
  EventBridge schedule and RAMP's 15:55 ET rebalance. Re-derive if either
  changes.
- **NYSE holidays** will false-positive `IBGatewayDown`, since
  `IBKRBroker.is_market_open()` is weekday+clock only.
