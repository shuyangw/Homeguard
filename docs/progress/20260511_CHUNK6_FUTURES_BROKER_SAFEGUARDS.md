# Chunk 6 (combined 6c-6k): Futures Broker Safeguards -- 2026-05-11

## Summary

Chunk 6 of the futures-focused Phase 0+1 implementation per `docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md`. Delivers 7 safeguard components plus the final IBKRFuturesBroker integration and a paper-trading validation runbook. The futures broker now has the boundary discipline, expiration awareness, margin pre-check, combo atomicity, per-cycle reconciliation, JSONL audit log, and roll-detection that the safeguards spec requires before any strategy may submit a live futures order.

Sub-chunks 6a (`FuturesPosition` + v3 schema) and 6b (`FuturesTradingInterface` ABC + skeleton) were documented separately in `20260511_CHUNK6a_FUTURES_POSITION_MODEL.md` and `20260511_CHUNK6b_FUTURES_BROKER_SKELETON.md`. This doc covers 6c-6k.

## Sub-chunks delivered

| Sub-chunk | Spec section | Component | Commit |
|---|---|---|---|
| 6c | §2.7 | `AuditLog` (JSONL daily rotation, append-only) | `a1c97f6` |
| 6d | §2.2 | `FuturesSymbolResolver` (continuous-to-per-contract translation) | `b1284b6` |
| 6e | §2.3 | `ExpirationGuard` (4 verdicts, 53-family thresholds) | `69c2e04` |
| 6f | §2.5 | `MarginGuard` (30% cash buffer, overnight 2x check) | `0e05667` |
| 6g | §2.4 | `FuturesComboOrderBuilder` (calendar roll + inter-commodity spread) | `1ee2035` |
| 6h | §2.6 | `PerCycleReconciler` (drift detection + gating + notifier callback) | `1942998` |
| 6i | §2.8 | `FuturesRollManager.get_upcoming_rolls` (real impl replacing v1 stub) | `25f938a` |
| 6j | integration | Wire safeguard chain into `IBKRFuturesBroker.submit_resolved_order` | `d922a4b` |
| 6k | §5 | Paper validation runbook + smoke test skeleton | `4c423ec` |

## Files Changed (Chunk 6c-6k)

- `src/trading/futures/audit_log.py` (new) -- `AuditLog` + `AuditEntry`
- `src/trading/futures/symbol_resolver.py` (new) -- `FuturesSymbolResolver` + `ContractResolution` + `ResolvedOrder` + `InvalidIntentError`
- `src/trading/futures/expiration_guard.py` (new) -- `ExpirationGuard` + `ExpirationVerdict` + `EXPIRATION_THRESHOLDS`
- `src/trading/futures/margin_guard.py` (new) -- `MarginGuard` + `MarginVerdict` + `MarginCheckResult`
- `src/trading/futures/combo_orders.py` (new) -- `FuturesComboOrderBuilder` + `ComboOrderSpec` + `ComboLegSpec` + `ComboOrderRejected`
- `src/trading/futures/reconciliation.py` (new) -- `PerCycleReconciler` + `ReconciliationVerdict` + `PositionDiff` + `ReconciliationResult`
- `src/trading/futures/roll_manager.py` (new) -- `FuturesRollManager` + `RollEvent`
- `src/trading/brokers/ibkr/ibkr_futures_broker.py` (modified) -- adds `submit_resolved_order`, `_ensure_safeguards`, `_ibkr_submit_stub`, `OrderRejectedError`; constructor now accepts injected safeguards
- `scripts/trading/futures_paper_smoke_test.py` (new) -- end-to-end smoke test of safeguard chain
- `docs/trading/futures_paper_validation_runbook.md` (new) -- 14-day paper-trading procedure with Day-14 acceptance checklist
- `tests/trading/futures/test_audit_log.py`, `test_symbol_resolver.py`, `test_expiration_guard.py`, `test_margin_guard.py`, `test_combo_orders.py`, `test_reconciliation.py`, `test_roll_manager.py` -- per-component test suites
- `.gitignore` -- adds `!docs/trading/` exception

## Design Notes (per sub-chunk)

### 6c AuditLog
- JSONL daily file rotation (`audit_YYYYMMDD.jsonl`); append-only, no in-place edits
- Helper methods `log_submission`, `log_fill`, `log_cancel`, `log_reject` -- callers don't construct `AuditEntry` directly
- Forensic-grade: every line a complete event with timestamp, strategy, symbol, action, order params, broker response

### 6d FuturesSymbolResolver
- `strategy_intent` must match `^[A-Z0-9]+\.v\.\d+$` -- enforces boundary between strategy logic (continuous) and broker (per-contract). Raw raw symbols cannot reach the broker.
- `resolve_active_contract` is cached per (symbol_root, as_of_date); `resolve_for_order` produces a `ResolvedOrder` frozen dataclass with everything `submit_resolved_order` needs
- Future hook: `_lookup_active_contract_from_definitions` will read `H:/Stock_Data/futures_definitions/year=Y/month=M/data.parquet` once that integration lands (sub-chunk 6l).

### 6e ExpirationGuard
- 4-verdict enum: OK / WARN / MUST_ROLL_OR_CLOSE / EXPIRED
- 53-family `EXPIRATION_THRESHOLDS` dict: ES=5, ZT=2, livestock LE/HE=5, ag ZC/ZS/ZW=3, etc.
- Time source is owned by the guard (`_today_fn`), NOT the position dataclass -- isolating tests from real-time drift
- `check_new_entry_with_expiration` blocks new entries when expiration is within threshold + horizon

### 6f MarginGuard
- `CASH_BUFFER_PCT = 0.30` -- reject if post-order free cash would drop below 30% of net liquidation
- `OVERNIGHT_INITIAL_MULTIPLIER = 2.0` -- reject if overnight margin (initial * 2.0) would exceed equity
- Calls `broker.what_if_order` (does NOT submit) + `broker.get_margin_status` -- both abstract methods on FuturesTradingInterface
- Returns `MarginCheckResult` with explicit reasoning -- caller can log it to audit

### 6g FuturesComboOrderBuilder
- `ComboOrderRejected` exception ban: if a combo is rejected by IBKR, do NOT fall back to separate legs. The whole point of the BAG order is SPAN spread credit + atomic exchange-side execution.
- `build_calendar_roll(short_month, long_month, quantity)` for rolls
- `build_inter_commodity_spread(legs)` for cross-product spreads

### 6h PerCycleReconciler
- 6 verdicts: MATCH / DRIFT_QUANTITY / DRIFT_PRICE / MISSING_ON_BROKER / MISSING_IN_STATE / EXPIRATION_DISAPPEARED
- `reconcile_and_gate(strategy) -> bool` -- returns False on drift, calls notifier callback (Discord alert in production)
- Strategies MUST call this at the start of every cycle. The boolean return is a hard gate.

### 6i FuturesRollManager.get_upcoming_rolls
- Replaces the v1 stub at `src/data/roll_detector.py` (kept for backward compat)
- Lookahead-days parameter; returns `RollEvent` per position needing roll within window
- Each event has `suggested_new_month` + `suggested_action` (ROLL_FORWARD / CLOSE / WARN_ONLY)

### 6j IBKRFuturesBroker integration
- Constructor accepts injected `audit_log`, `expiration_guard`, `margin_guard` -- testable without globals
- `_ensure_safeguards` lazy-instantiates defaults if not injected
- `submit_resolved_order(resolved_order, expiration_date, hold_overnight)` runs the chain:
  1. ExpirationGuard.check_new_entry_with_expiration -> raises OrderRejectedError on MUST_ROLL_OR_CLOSE/EXPIRED
  2. MarginGuard.pre_trade_check -> raises OrderRejectedError on REJECT/REJECT_OVERNIGHT
  3. AuditLog.log_submission -> JSONL append
  4. _ibkr_submit_stub -> synthetic order ID (real ib_async integration deferred to 6l)

### 6k Paper validation
- 14-day paper-trading window is irreducible -- the runbook documents the procedure; the smoke test script validates day-0 prerequisites in ~30 seconds
- Smoke test is the futures analogue of `scripts/trading/smoke_test_ibkr_paper.py`
- Day-14 acceptance checklist explicitly maps to safeguards spec sections 2.1-2.8

## Commits

- `a1c97f6` feat(trading): AuditLog with JSONL daily rotation for forensic forensics
- `b1284b6` feat(trading): FuturesSymbolResolver maps continuous intent to per-contract order
- `69c2e04` feat(trading): ExpirationGuard with per-family thresholds and 4 verdicts
- `0e05667` feat(trading): MarginGuard with 30% cash buffer + overnight check
- `1ee2035` feat(trading): FuturesComboOrderBuilder for calendar rolls + inter-commodity spreads
- `1942998` feat(trading): PerCycleReconciler with drift gating + notifier callback
- `25f938a` feat(trading): real FuturesRollManager.get_upcoming_rolls (replaces v1 stub)
- `d922a4b` feat(trading): wire safeguard chain into IBKRFuturesBroker.submit_resolved_order
- `4c423ec` docs(trading): futures paper validation runbook + smoke test skeleton

## Validation

- Each sub-chunk shipped with its own test suite (per-component coverage). Combined: 9 test files under `tests/trading/futures/`.
- The IBKRFuturesBroker contract test from sub-chunk 6b still passes (parametrized over `REQUIRED_FUTURES_METHODS`).
- The smoke test script `python scripts/trading/futures_paper_smoke_test.py` is callable but the real round-trip requires `_ibkr_submit_stub` to be replaced with real ib_async integration (sub-chunk 6l).

## Known Issues / Remaining Work

- **Sub-chunk 6l deferred**: real `ib_async` integration in `IBKRFuturesBroker._ibkr_submit_stub` plus the parallel `cancel_order` / `get_order` / `get_futures_positions` / `get_margin_status` / `what_if_order` paths. ~200-300 LOC. The runbook documents the code sketch.
- **`futures_definitions/` lookup not wired**: `FuturesSymbolResolver._lookup_active_contract_from_definitions` is a placeholder. Needs a `DefinitionsLoader` class reading from `H:/Stock_Data/futures_definitions/year=Y/month=M/data.parquet`. Without this, expiration dates fall back to placeholders.
- **No strategy uses these safeguards yet**. The first futures strategy to wire `submit_resolved_order` into its execution path will exercise the safeguards end-to-end. Until then, the chain is dormant.
- **State manager (`src/trading/state/strategy_state_manager.py`) still on v2 schema**. The migration script ships but production state hasn't been migrated. This is fine because no futures position is live yet -- migration is part of futures-strategy deployment day.

## Decision Gate

PROCEED to Chunk 7 (deferred signal pipelines per master spec §4 Chunk 7):
- VIX-equivalent from ES.OPT or ES realized vol -> `derive_vix_equivalent(date)` under `src/data/derivations/futures/`
- FOMC/NFP/CPI hardcoded YAML calendars at `config/macro_calendar/`
- OI aggregate from `futures_statistics/`

## Reproduction Commands

```bash
cd C:/Users/qwqw1/Dropbox/cs/github/Homeguard
conda run -n fintech pytest tests/trading/futures/ -v
# Expected: per-component test suites all green (count varies by sub-chunk -- see individual sub-chunk validation sections)
```
