# Close Out 2026-04-23 RAMP Rebalance Failure Audit - 2026-04-24

## Summary
Followed up on the 2026-04-23 RAMP rebalance that placed zero orders. The proximate cause (`IBKRBroker` missing `get_open_orders` -> portfolio health check caught AttributeError -> entry blocked) was fixed by `665a35f` the prior evening. The audit surfaced four additional items: the gateway service file on EC2 had drifted from the repo (still had `Restart=on-failure` despite the repo being changed to `Restart=always` a week earlier in `35e3260`), `CoinbaseBroker` existed in the tree but was never registered in the factory (silent routing fallback to Alpaca), `MKR/USD` was in CSCM's live universe but unsupported on the Alpaca failover broker (hourly Binance fetch failure for 24+ hours), and the startup banner was logging CLI args as if they were effective runtime config. All four are closed out in this change.

## Changes Made

### Ops: gateway service file redeployed on EC2 (no repo change)
- Copied `infra/ec2/services/homeguard-gateway.service` -> `/etc/systemd/system/homeguard-gateway.service` on EC2 via SSM, daemon-reload, start.
- Before: `Restart=on-failure`, `RestartSec=30`, no StartLimit directives. After: `Restart=always`, `RestartSec=60`, StartLimit directives present.
- Verified: `ss -ltn | grep 4002` shows listener within 2s, `IBC: Login has completed` in journal, `systemctl is-active homeguard-gateway` = `active`.
- This is the fix that keeps IB's nightly forced logout from permanently killing the gateway -- prior behavior left the unit dead until next EC2 boot, which was blocking any trade on a day where the instance was not stop/started overnight.

### `src/trading/brokers/broker_factory.py`
- Added a `coinbase` / `cb` branch to `BrokerFactory.create_broker()` between the IBKR and TD Ameritrade branches. Reuses the already-production-ready `CoinbaseBroker` class (`src/trading/brokers/coinbase_broker.py`), which reads `COINBASE_API_KEY`/`COINBASE_API_SECRET` from env if not passed in config.
- Updated `list_supported_brokers()` to include `'coinbase'` and the `ValueError` message so unsupported-broker errors are accurate.
- **Side effect**: the `[!] [Routing] Strategy 'cscm' references unknown broker 'coinbase', using default` warning from `src/trading/config/broker_routing.py:118-121` goes away. CSCM's actual execution path is unchanged -- `CSCMLiveAdapter` has always bypassed `broker_routing.yaml` via `CryptoBrokerRouter` -- so this is correctness hygiene for anyone who later queries the factory directly, not a live behavior change.

### `config/trading/cscm_live.yaml`
- Removed `MKR/USD` from `universe`. 13 symbols -> 12 symbols.
- Added YAML comment documenting why (not on Alpaca, failing hourly on Binance) with a pointer back to this file.
- Not replaced with CRV/GRT (which Amendment 8 in `docs/strategies/20251230_CSCM_OPTIMIZATION_RESULTS.md` documents as the intended substitutes) -- deferred. If desired later, the CSCMLiveAdapter default universe already includes CRV/GRT, so config can simply be shortened further.

### `scripts/trading/run_live_paper_trading.py` (startup banner)
- Lines 1250-1254: relabeled `Universe:`, `Position size:`, `Max positions:` to `CLI universe arg:` / `CLI position size:` / `CLI max positions:` and added a note that the strategy runtime may override them.
- Motivation: banner was printing `Universe: faang (5 symbols) / Position size: 5.0% / Max positions: 3` for RAMP while the actual 15:55 ET run used the S&P 500 universe with `TopN=10` and 1/N sizing from the regime detector. That made debugging the 2026-04-23 non-trade harder than it should have been.

## Commits
- `<filled in after commit>` fix(ops): redeploy gateway config, register coinbase, drop MKR, clarify startup log

## Known Issues / Remaining Work
- **Repo service file has a latent bug**: `infra/ec2/services/homeguard-gateway.service` puts `StartLimitBurst=5` and `StartLimitIntervalSec=600` in the `[Service]` section, but systemd requires these in `[Unit]`. They are silently ignored. Not blocking (we still have `Restart=always`), but the rate-limit backstop the author intended is not actually in effect. Same check should be done on every other unit under `infra/ec2/services/`.
- **Deployment-drift audit is broader than one file**: this incident shows the `install_ibkr_gateway.sh` path does not enforce a re-copy of an already-present unit, and there's no general "sync /etc/systemd/system/ to infra/ec2/services/" command. Worth a short script that can be rerun after any service-file change: `for f in infra/ec2/services/*.service; do diff -q "$f" "/etc/systemd/system/$(basename $f)"; done` at minimum, and a systemctl daemon-reload/restart for anything that diverged.
- **2026-04-22 crash loop**: 179+ restarts in ~1.7 hours due to `OMRLiveAdapter.__init__() missing 1 required keyword-only argument: 'broker_name'`. Masked now by `--strategy ramp` pin in `homeguard-multi.service`, and all current call sites (`run_live_paper_trading.py` + all `tests/` files) correctly pass `broker_name`. No regression test exists that would have caught the original omission at construction time -- a cheap unit test that imports `OMRLiveAdapter` and instantiates with mocks would close this hole.
- **Dual crypto routing**: `src/trading/brokers/broker_factory.py` now knows about Coinbase, but `CSCMLiveAdapter` still uses `CryptoBrokerRouter` directly. Two code paths for the same decision. Unifying them is a medium-sized refactor; out of scope here but worth planning.
- **EventBridge schedule margin**: stop-Lambda fires at 16:30 ET, RAMP rebalance at 15:55 ET, market close at 16:00 ET. 30 min between rebalance and stop is tight if the broker is slow to confirm fills. Not a bug today, but a fragility to watch.
- **MKR replacement TBD**: per Amendment 8, CRV and GRT are the intended replacements. Deferred until the CSCM rebalance-never-fires issue (logged 2026-04-23) is resolved -- no point tuning the universe when the rebalance itself is broken.

## Validation
On EC2 (via SSM) post-deploy:
- `diff <(sudo cat /etc/systemd/system/homeguard-gateway.service) /home/ec2-user/Homeguard/infra/ec2/services/homeguard-gateway.service` -- empty.
- `sudo systemctl is-active homeguard-gateway` -- `active`.
- `sudo ss -ltn | grep 4002` -- listener present.
- `sudo journalctl -u homeguard-gateway -n 50 | grep "Login has completed"` -- matched.
- `sudo journalctl -u homeguard-multi -n 200 | grep "references unknown broker"` -- empty after restart.
- `sudo journalctl -u homeguard-cscm -n 200 | grep "Binance.*MKR"` -- empty after restart.
- `sudo journalctl -u homeguard-multi -n 80 | grep "CLI universe arg"` -- confirms new log wording.
- RAMP broker no longer emits `[X] [IBKR] Failed to get account: Not connected` after the multi-service restart reattaches to the new gateway session.

Local:
- `python -c "from src.trading.brokers.broker_factory import BrokerFactory; print(BrokerFactory.list_supported_brokers())"` includes `'coinbase'`.
