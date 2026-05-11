# Futures Paper Trading Validation Runbook

**Purpose**: 14-day paper-trading observation window required before any futures strategy moves from paper to real money. This is the acceptance gate from `docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md` Section 5.

**Owner**: Shuyang.
**Created**: 2026-05-11 (sub-chunk 6k).

---

## 1. Prerequisites

Before starting the 14-day window, all of the following must be true:

1. **All sub-chunks 6a-6k merged to main** (DONE):
   - 6a: `FuturesPosition` + v3 schema (`docs/progress/20260511_CHUNK6a_FUTURES_POSITION_MODEL.md`)
   - 6b: `FuturesTradingInterface` ABC + `IBKRFuturesBroker` skeleton (`docs/progress/20260511_CHUNK6b_FUTURES_BROKER_SKELETON.md`)
   - 6c-6k: 7 safeguard components + paper runbook (`docs/progress/20260511_CHUNK6_FUTURES_BROKER_SAFEGUARDS.md`)

2. **Real `ib_async` integration in `IBKRFuturesBroker`** (DONE, merged `9c9cc95`, fixes `5b2925c`):
   `_ibkr_submit`, `cancel_order`, `get_order`, `get_orders`, `get_open_orders`, `get_futures_positions`, `get_futures_position`, `close_futures_position`, `close_all_futures_positions`, `what_if_order`, `get_margin_status`, `place_futures_order`, `place_futures_combo_order`, `get_latest_trade` are all wired against `ib_async` via the existing `IBKRConnectionManager` singleton. All async round-trips go through `run_sync(...Async(...))`. Validated end-to-end on EC2 paper Gateway 2026-05-11 (see `docs/progress/20260511_FUTURES_EC2_PAPER_SMOKE_VALIDATION.md`).

3. **`futures_definitions/` integration** (DONE, merged `3c757e1`):
   `FuturesDefinitionsLoader` (`src/data/futures_definitions_loader.py`) reads `<storage>/futures_definitions/year=Y/month=M/data.parquet`, returns real expirations. Wired through `FuturesSymbolResolver` -> `ResolvedOrder.expiration_date` -> `submit_resolved_order` -> `ExpirationGuard`. Falls back to `ib.reqContractDetailsAsync` when local partitions aren't deployed (used on EC2 where the data is not synced).

4. **IBKR Gateway running on port 4002 (paper)**:
   ```
   IBKR Gateway -> Paper Trading -> Connect
   Futures permissions enabled in account settings (CME / NYMEX / COMEX / CBOT)
   ```
   On EC2 (`homeguard-trading` instance), the running `homeguard-multi.service` already binds Gateway on 4002 with clientId=10. The smoke test uses clientId=99 so it doesn't collide.

5. **Audit log directory writable**: `~/.homeguard/audit/` for production strategies, `~/.homeguard/audit_smoke/` for smoke runs.

6. **Smoke test passes**: `python scripts/trading/futures_paper_smoke_test.py` exits 0 with all `[+]` markers. **PASSED on EC2 2026-05-11** (see validation log).

---

## 2. The 14-day procedure

### 2.1 Day 0 (setup)

1. Start the strategy in paper mode with a small position (e.g. 1 MES contract):
   ```bash
   python scripts/trading/run_live_paper_trading.py --strategy adaptation_d_paper
   ```
2. Confirm initial reconciliation passes; strategy state matches IBKR paper positions.
3. Verify Discord alerts wired and firing on test (`MarginGuard` rejection trigger via a deliberately too-large order, then revert).
4. Take a snapshot of: account equity, initial margin, free cash. Save as `paper_validation_day0.json`.

### 2.2 Days 1-14 (daily observations)

Every trading day at 4pm ET (after market close, before overnight margin transition):

| Check | Pass criterion |
|---|---|
| Reconciliation drift | `PerCycleReconciler` returns MATCH at every cycle (no drift events in audit log) |
| Audit log integrity | Every order in IBKR portal matches an audit log entry (cross-check 5 random orders/day) |
| Margin headroom | `get_margin_status().free_cash / net_liquidation >= 0.30` at every market-hours check |
| Expiration alerts | If any position has `days_to_expiration <= threshold + 2`, Discord alert fired and roll was scheduled |
| Combo execution | Any roll executed as single BAG order (single fill timestamp per contract, no orphan legs in IBKR portal) |
| Position consistency | Every `(symbol_root, contract_month)` in `strategy_positions.json` matches IBKR; no phantom positions |

Record observations in `docs/progress/20260511_CHUNK6k_paper_validation_dayN.md` (one file per day, append-only).

### 2.3 Day 14 (acceptance review)

Cross-cutting acceptance criteria from safeguards spec Section 5:

```
[ ] §2.1 Position model with contract month identity
    [ ] strategy_positions.json v3 migration executed against production
    [ ] All 14 days of positions correctly identified by (symbol_root, contract_month)

[ ] §2.2 Symbol resolver
    [ ] Every order on every day used resolve_for_order; no raw symbols bypassed
    [ ] Resolved symbol matched IBKR's accepted contract on every order

[ ] §2.3 Expiration guard
    [ ] Discord alert fired on first WARN encounter per position
    [ ] No EXPIRED positions occurred (strategy rolled or closed in time)

[ ] §2.4 Combo atomicity
    [ ] All rolls executed as single BAG orders
    [ ] No separate-leg fallback paths in audit log
    [ ] SPAN spread credit visible in margin status during spread holding

[ ] §2.5 Margin guard
    [ ] No orders rejected at IBKR level (all rejections caught pre-trade)
    [ ] Overnight margin check fired on at least 3 different days
    [ ] No instances of margin exceeding 70% of equity

[ ] §2.6 Per-cycle reconciliation
    [ ] 0 drift events across 14 days (or all drifts correctly identified and resolved)
    [ ] reconcile_and_gate returned True at start of every cycle

[ ] §2.7 Audit log
    [ ] 100% of orders matched by 100% of fills (count audit submit entries vs IBKR fills)
    [ ] Daily file rotation verified (14 distinct audit_*.jsonl files)
    [ ] JSONL parse-clean: every line valid JSON

[ ] §2.8 get_upcoming_rolls
    [ ] At least 1 roll detected and executed during the window
    [ ] Scheduled daily job posted upcoming-roll summary to Discord

Cross-cutting:
[ ] No reconciliation drift across 14 days
[ ] No unexpected IBKR account warnings (margin call, position-limit breach, etc.)
[ ] All 8 unit test suites still pass after 14 days (no regressions from production data)
```

If any box is unchecked, the strategy stays in paper. Re-run the 14-day window after addressing the failure.

---

## 3. Acceptance gate

When all boxes above are checked, the strategy may move from paper to real-money live at the recommended initial allocation per `02_IMMEDIATE_NEXT_STEPS.md` Section 5.

**Do not skip this gate.** Paper trading is unrestricted (it's the validation environment); real money is gated.

---

## 4. Common failure modes and resolution

### 4.1 Reconciliation drift detected

- Read the audit log around the drift timestamp; identify the order or event that caused state/broker divergence.
- Common causes: IBKR auto-liquidation (margin breach), manual operator action via TWS, async fill that wasn't captured.
- Resolution: fix the strategy's reconciliation handling. Do NOT use `--force-cycle` in real money without investigating.

### 4.2 Margin guard rejected an expected order

- Inspect `what_if_order` output for the rejected order.
- Verify the broker's account summary matches expectations.
- If the rejection is correct (genuine margin shortfall), reduce position sizing.
- If the rejection is a false positive (margin guard parameters off), recalibrate `MarginGuard.CASH_BUFFER_PCT` or `OVERNIGHT_INITIAL_MULTIPLIER`.

### 4.3 Combo order rejected by IBKR

- Read the IBKR rejection message in the audit log.
- Common causes: exchange routing (after-hours combo not supported), inactive leg (one contract expired), SPAN credit not granted for the combination.
- Resolution: do NOT fall back to separate-leg orders. Investigate the rejection root cause; resubmit the combo or skip the trade.

### 4.4 Expiration cliff hit (EXPIRED verdict)

- This is an emergency. The position should have rolled or closed earlier.
- Check the daily job that runs `get_upcoming_rolls`; verify it was scheduled and ran.
- Manually flatten the position immediately.
- Add a safety net: cron job at 3pm ET each day to detect any position with `days_to_expiration <= 1` and alert.

---

## 5. IBKR integration -- shipped

Real `ib_async` integration is now live (merge `9c9cc95`, fixes `5b2925c`). All `FuturesTradingInterface` methods go through `IBKRConnectionManager.run_sync(...Async(...))`. Exchange routing is in `_EXCHANGE_BY_ROOT` (CME / NYMEX / COMEX / CBOT for 40+ symbol roots). Key implementation files:

- `src/trading/brokers/ibkr/ibkr_futures_broker.py` -- the broker
- `src/trading/futures/symbol_resolver.py` -- continuous-to-per-contract resolution
- `src/data/futures_definitions_loader.py` -- expiration source

End-to-end smoke validation against EC2 paper Gateway: `docs/progress/20260511_FUTURES_EC2_PAPER_SMOKE_VALIDATION.md`.

The remaining gap before the 14-day window can start is operational: a strategy adapter must invoke `broker.submit_resolved_order(resolved, hold_overnight=True)` for two weeks. That work belongs to Phase 2+ strategy adaptations.
