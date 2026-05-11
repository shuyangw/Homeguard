# Futures Paper Smoke Test -- EC2 Validation -- 2026-05-11

## Summary

Ran `scripts/trading/futures_paper_smoke_test.py` against the IBKR paper Gateway on the `homeguard-trading` EC2 instance. The full safeguard chain (`ExpirationGuard` -> `MarginGuard` -> `AuditLog` -> real `ib_async` submission) executed end-to-end, placing a real LIMIT BUY at 50% below market on MESM6 (June 2026 micro E-mini S&P), confirming visibility, cancelling, and verifying clean post-state. **Three bugs surfaced during the live run; all three fixed and re-validated.** Production `homeguard-multi` service (RAMP, clientId=10) was untouched throughout.

## Setup

- EC2: `homeguard-trading` (t4g.medium, ARM64)
- Service running: `homeguard-multi.service` (RAMP, clientId=10) -- 15h+ uptime, untouched
- Smoke test connected via clientId=99 (separate from the running service)
- IBKR Gateway: paper trading, port 4002
- Symbol: MES (micro E-mini S&P 500)
- Order: LIMIT BUY 1 contract @ 50% below market

## Bugs Found and Fixed (commit `5b2925c`)

### Bug 1: Sync ib_async methods hung indefinitely

**Symptom**: Step 0d (resolve via `reqContractDetails`) hung past 120s timeout (`EXIT=124`).

**Root cause**: `ib_async`'s synchronous methods (`ib.qualifyContracts`, `ib.reqContractDetails`, `ib.whatIfOrder`) internally call `self._run(coroutine)`, which works from within the IB event loop but blocks indefinitely when called from a different thread. Homeguard runs `IBKRConnectionManager` on a background asyncio loop; any external thread must dispatch via `run_sync(coroutine)`.

**Fix**: Replace sync calls with their `Async` variants wrapped through `run_sync`:
- `_build_future_contract`: `qualifyContracts` -> `run_sync(qualifyContractsAsync(c))`
- `what_if_order`: `whatIfOrder` -> wrap `whatIfOrderAsync` in a local `async def`, then `run_sync`
- Smoke test `_resolve_intent_via_ibkr`: same wrapper pattern for `reqContractDetailsAsync`

Note: `qualifyContractsAsync` is declared `async def` (returns coroutine), but `reqContractDetailsAsync` and `whatIfOrderAsync` are `def` returning `Awaitable[X]` (Future, not coroutine). `run_sync` strictly requires a coroutine, so the latter two need wrapping in an `async def` shim.

### Bug 2: MarginGuard field name mismatch

**Symptom**: Step 2a failed with `KeyError: 'maintenance_margin_after'`.

**Root cause**: `MarginGuard.pre_trade_check` reads `whatif["maintenance_margin_after"]` and `whatif["initial_margin_after"]` (absolute post-trade values), but `IBKRFuturesBroker.what_if_order` was returning `initial_margin` / `maintenance_margin` keyed off `OrderState.initMarginChange` (the delta, not the absolute).

**Fix**: Updated `what_if_order` to return both naming conventions:
- `initial_margin_after`, `maintenance_margin_after`, `equity_with_loan_after` -- parsed from `OrderState.{init,maint,equityWithLoan}After` (what MarginGuard reads)
- `initial_margin_change`, `maintenance_margin_change` -- the deltas (for logging)
- `initial_margin` -- alias for change (smoke test still uses it)
- All five fields parse `OrderState`'s string-typed values defensively (ib_async returns `str` for margin tags, not `float`)

### Bug 3: EC2 doesn't have local futures_definitions data

**Symptom**: Step 0d's primary path (`FuturesDefinitionsLoader.get_expiration`) raised `ValueError: no active contract for MES on 2026-05-11`, indirectly because `H:/Stock_Data/futures_definitions/` (Homeguard data root) doesn't exist on EC2 -- the running production strategies (RAMP, OMR, CSCM) don't need futures data.

**Fix**: Added a fallback path `_resolve_intent_via_ibkr` (commit `7b6f862`) that queries `ib.reqContractDetailsAsync` directly. Picks the contract whose `lastTradeDateOrContractMonth` is the earliest >= today. Returns a `ResolvedOrder` populated entirely from IBKR's own data -- no local partitions needed.

This means the smoke test (and any future operator script) can run on machines that don't carry the futures data partitions. The local-data path is still preferred when available (faster, no API round-trips).

## End-to-end Result

```
======================================================================
FUTURES PAPER SMOKE TEST
  symbol_root=MES  qty=1  port=4002  clientId=99
======================================================================

===== step 0a: connect to IBKR paper (port=4002, clientId=99) =====
  [+] connected to IBKR paper

===== step 0b: fetch account margin status =====
  [+] net_liq=$1,016,872.47 avail=$984,242.98 init_margin=$32,072.55

===== step 0c: fetch baseline futures positions =====
  [+] baseline: 0 futures positions

===== step 0d: resolve MES.v.0 with real expiration =====
[!]   local data unavailable (ValueError); falling back to IBKR contract details
  resolved via IBKR reqContractDetails
  [+] MES.v.0 -> MESM6 (contract_month=202606, expiration=2026-06-18)

===== step 0e: fetch reference price for MESM6 =====
  [+] MESM6 reference price = $7,433.25

===== step 1: what_if_order pre-trade margin estimate =====
  [+] estimate: init_margin=$2,344.00 commission=$0.6200

===== step 2a: LIMIT BUY 1 MESM6 @ 50% below market =====
  placing LIMIT BUY 1 MESM6 @ $3716.50 (last $7433.25)
[IBKR-FUT] Submitted MESM6 BUY 1 -> orderId=52 status=Submitted
  [+] submitted orderId=52 status=pending

===== step 2b: get_order + confirm visibility =====
  [+] get_order(52) -> status=pending

===== step 3a: cancel_order(52) =====
[IBKR-FUT] Cancelled order 52
  [+] cancel_order(52) -> True
  [+] post-cancel status=cancelled

===== step 4a: verify clean state =====
  [+] positions unchanged: 0 futures positions
  [+] no lingering open orders from this run

===== step 4b: verify audit log captured events =====
  [+] audit log has 2 events for this run: ['submit', 'cancel']

===== step Z: disconnect IBKR cleanly =====
  [+] broker.stop() completed

=== FUTURES SMOKE TEST PASSED (MES) ===
```

## What This Validates

The full live-paper plumbing for futures, end-to-end:

| Layer | Validated by |
|---|---|
| Connection lifecycle | step 0a (connect) and step Z (clean disconnect) |
| Account-summary reads | step 0b ($1,016,872 net_liq, $32,072 init margin from RAMP positions) |
| Position reads | steps 0c, 4a |
| Continuous -> per-contract resolution | step 0d, fallback path via IBKR |
| Real expiration lookup | step 0d (2026-06-18 for MESM6 -- correct CME E-mini Jun 2026 expiry) |
| Market-data snapshot | step 0e ($7,433.25 delayed quote) |
| Pre-trade margin estimate | step 1 ($2,344 init -- accurate for 1 MES on a ~$7,400 contract) |
| ExpirationGuard | implicit in step 2a (would have rejected if Jun 2026 was inside threshold; passed) |
| MarginGuard | implicit in step 2a (would have rejected if 30% cash buffer breached; passed) |
| AuditLog -- submit | step 4b (entry present in JSONL) |
| `_ibkr_submit` -> real IBKR | step 2a (orderId=52 from IBKR, not a synthetic) |
| Order visibility | step 2b (get_order returns the pending order) |
| Cancellation | step 3a (cancel_order True, post-cancel status `Cancelled`) |
| AuditLog -- cancel | step 4b (`log_cancel` written) |
| Clean state verification | step 4a (no new positions, no lingering opens) |

## Commits

- `7b6f862` feat(trading): smoke test falls back to IBKR contract details when local data missing
- `5b2925c` fix(trading): async wrappers + margin field names for live IBKR paper

(Both pushed to `origin/main` 2026-05-11.)

## RAMP Service Status Throughout

```
homeguard-multi.service - Homeguard Multi-Strategy Trading Bot (pinned to RAMP)
   Active: active (running) since Sun 2026-05-10 23:14:46 UTC
   Main PID: 2071 (homeguard-ramp)
   IBKR clientId: 10 (separate from smoke test's 99)
```

Service uptime preserved (no restart). RAMP trades S&P 500 stocks via the same paper Gateway; the smoke test placed a futures order on a different instrument with a different clientId. No collision, no orphan orders, no margin impact on RAMP's positions.

## Known Gaps / Follow-ups

- **No live ESG/futures data on EC2**: The IBKR fallback path is sufficient for production use (strategies query IBKR for active contracts anyway), but if a futures strategy needs historical data on EC2, a separate sync step (rsync or S3 backed) is required.
- **`min_price_increment_amount` field still mis-scaled** in `ContractDefinition` (ES reports `0.125` vs real `$12.50`). Not blocking -- the field isn't consumed by the safeguard chain. Defer until a strategy needs to convert ticks to dollars.
- **One unawaited-coroutine pytest warning** in `test_get_margin_status_parses_account_values`. Benign (mock test path), not a runtime issue.

## Acceptance Gate

Both technical blockers for the 14-day paper validation window are resolved:
- Blocker A: real `ib_async` integration -- DONE
- Blocker B: real expiration dates -- DONE (with IBKR fallback for EC2)

The remaining blocker is operational: wire a strategy to call `submit_resolved_order` and run it in paper for two weeks. That's Phase 2+ adaptation work.
