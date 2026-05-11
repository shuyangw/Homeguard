# Futures Paper Trading Validation Runbook

**Purpose**: 14-day paper-trading observation window required before any futures strategy moves from paper to real money. This is the acceptance gate from `docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md` Section 5.

**Owner**: Shuyang.
**Created**: 2026-05-11 (sub-chunk 6k).

---

## 1. Prerequisites

Before starting the 14-day window, all of the following must be true:

1. **All sub-chunks 6a-6j merged to main**:
   - 6a: `FuturesPosition` + v3 schema (`docs/progress/20260511_CHUNK6a_*`)
   - 6b: `FuturesTradingInterface` ABC + `IBKRFuturesBroker` skeleton (`docs/progress/20260511_CHUNK6b_*`)
   - 6c-6j: All 7 safeguard components

2. **Real IBKR API integration in `IBKRFuturesBroker._ibkr_submit_stub`**:
   The current `_ibkr_submit_stub` method in `src/trading/brokers/ibkr/ibkr_futures_broker.py` returns synthetic order IDs. Before validation, replace it with real `ib_async` calls:
   - Instantiate `ib_async.IB` connection per `IBKRConfig`
   - Build `Future` contract for the resolved `(symbol_root, contract_month)`
   - Call `ib.qualifyContracts(future)` to bind the contract
   - Call `ib.placeOrder(future, order)` to submit
   - Return `{"orderId": trade.order.orderId, "permId": trade.order.permId, "status": str(trade.orderStatus.status)}`
   - Wire `cancel_order`, `get_order`, `get_futures_positions`, `get_margin_status`, `what_if_order` similarly
   This is approximately 200-300 LOC of additional integration code.

3. **`futures_definitions/` integration in symbol resolver**:
   The current `FuturesSymbolResolver` doesn't read expiration dates from `futures_definitions/`. Add a `get_expiration(symbol_root, contract_month) -> date` method on a new `DefinitionsLoader` class that reads from `H:/Stock_Data/futures_definitions/year=Y/month=M/data.parquet` filtered to the contract. Use this in the smoke test instead of the placeholder date.

4. **IBKR Gateway running on port 4002 (paper)**:
   ```
   IBKR Gateway -> Paper Trading -> Connect
   Futures permissions enabled in account settings
   ```

5. **Audit log directory writable**: `~/.homeguard/audit/` exists and the running process has write permission.

6. **Smoke test passes**: `python scripts/trading/futures_paper_smoke_test.py` exits 0 with all `[OK]` markers.

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

## 5. Real IBKR integration deferred

This runbook assumes `_ibkr_submit_stub` has been replaced with real `ib_async` calls. That integration is approximately 200-300 LOC and was deferred from sub-chunk 6j. Code sketch:

```python
def _ibkr_submit(self, resolved: ResolvedOrder) -> dict[str, Any]:
    from ib_async import Future, LimitOrder, MarketOrder
    self._ensure_connection()
    contract = Future(
        symbol=resolved.symbol_root,
        lastTradeDateOrContractMonth=resolved.contract_month,
        exchange="GLOBEX",
    )
    self._ib.qualifyContracts(contract)
    if resolved.order_type == OrderType.LIMIT:
        order = LimitOrder(
            action=resolved.side.value,
            totalQuantity=resolved.quantity,
            lmtPrice=resolved.limit_price,
            tif=resolved.time_in_force.value,
        )
    else:
        order = MarketOrder(
            action=resolved.side.value,
            totalQuantity=resolved.quantity,
            tif=resolved.time_in_force.value,
        )
    trade = self._ib.placeOrder(contract, order)
    self._ib.sleep(0.5)  # let IBKR acknowledge
    return {
        "orderId": trade.order.orderId,
        "permId": trade.order.permId,
        "status": str(trade.orderStatus.status),
    }
```

A separate sub-chunk (call it 6l) should land this before the smoke test can run for real.
