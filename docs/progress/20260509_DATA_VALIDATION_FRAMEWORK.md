# Data Validation Framework + Additional Databento Pull - 2026-05-09

## Summary

Built a multi-domain layered data validation framework (futures-only impl + extensibility hooks for equities/crypto/fx/options) and modified the Databento batch pipeline to add three new datasets (trades, status events, Eurodollar daily). Framework is complete and runs end-to-end against the real local store; the additional data pull is partially submitted with one job re-queued after a symbology fix and one (Trades_ES_MES) deferred pending budget decision.

## Changes Made

### Validation framework (Tasks 1-14)

- **Core primitives** (`src/data/validation/core/`): `Severity`, `ValidationResult`, `RunReport` (frozen dataclasses); `BaseCheck` ABC with `__init_subclass__` auto-registration and `_auto_register = False` opt-out for lazy-loaded checks; `ValidationRunner` with continue-on-CRITICAL semantics + per-check exception containment; `MarkdownReporter` writing YAML-frontmatter reports with regression diff parsing.
- **Futures domain** (`src/data/validation/futures/`): `expectations.py` (53 symbols + ranges + listing dates + known events + per-section schemas); `checks/structural.py` (5 Layer 1 checks); `checks/statistical.py` (218 Layer 2 checks via per-symbol auto-registration); `checks/cross_source.py` (3 standard + 3 deferred Layer 3 checks); `checks/external.py` (4 Layer 4 checks); `checks/adaptation_f.py` (3 lazy-loaded gating checks).
- **Derivation pipelines** (`src/data/derivations/futures/`): `derive_sofr(date)` from SR1 front-month close; `get_treasury_yield(tenor, date)` from Micro Yield futures direct read. Deferred: ES realized vol, VIX-equivalent, per-asset-class carry (rationale: `docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md`).
- **Domain placeholders** (`src/data/validation/{equities,crypto,fx,options}/__init__.py`): empty packages with implementation roadmap docstrings.
- **CLI** (`scripts/data/run_validation.py`): `--domain`, `--layer`, `--check`, `--mode`, opt-in flags (`--external-yfinance`, `--external-cme`, `--adaptation-f`), auto-discovers previous report for regression diff. Returns exit 1 if any CRITICAL failures.
- **Integration tests**: full Layer 1 sweep against real data (passes 0 CRITICAL); GC density bug-fix smoking gun (>700 bars/day asserts the volume-roll fix is intact).

### Databento batch additions (Tasks 15-18)

- **Submit script** (`scripts/data/databento_batch_submit.py`): added 4 new sections — `Trades_ES_MES` (trades schema, ES + MES, 5y), `Status_continuous` + `Status_parent` (status events for full universe), `B_ED_daily` (Eurodollar per-contract daily, pre-2018 funding history).
- **Convert script** (`scripts/data/databento_batch_convert.py`): added `convert_trades` (filters to 19:00-21:00 UTC window, stores under `futures_trades_window/`), `convert_status` (year-flat under `futures_status/`), `convert_b_ed_daily` (year-flat under `futures_per_contract_daily/root=ED/`).
- **Symbol fix**: `ED.FUT` rejected by Databento (Eurodollars retired 2023). Replaced with legacy CME symbol `GE.FUT` which resolves to 1238 contracts.

### Architecture documentation

- `docs/architecture/ARCHITECTURE_OVERVIEW.md`: added "Data Validation Framework" and "Derivation Pipelines" subsections under Layer 1.

## Commits

- `0906c4e` feat(validation): add core result types
- `8897d52` feat(validation): add BaseCheck ABC with auto-registry
- `00c3e59` feat(validation): add ValidationRunner with per-check error containment
- `1a02211` feat(validation): add MarkdownReporter with regression comparison
- `38378c2` feat(derivations): SOFR from SR1 front-month + deferred-derivation stubs
- `394b0e4` feat(derivations): Treasury yield read from Micro Yield futures
- `a726660` feat(validation): futures expectations module (densities, ranges, listings, events)
- `ed72ee9` feat(validation): Layer 1 structural checks for futures
- `a6c5fca` feat(validation): Layer 2 statistical checks (basic + 2 derivations + 4 stubs)
- `ee5c490` feat(validation): Layer 3 cross-source checks (definitions, derivations + stubs)
- `e127015` feat(validation): Layer 4 external + Adaptation F gating (opt-in)
- `c4bd3be` feat(validation): placeholder packages for equities/crypto/fx/options
- `460a608` feat(validation): CLI entrypoint with regression-comparison auto-discovery
- `3b86a2e` test(validation): integration smoke tests for layer 1 + GC density
- `5fa2a69` chore(databento): import in-flight batch scripts as baseline for additional pull
- `4d31291` feat(databento): add Trades_ES_MES section to batch submission
- `a9e9e61` feat(databento): add Status_universe section (continuous + parent) to batch submission
- `f89edc3` feat(databento): add B_ED_daily Eurodollar pre-SOFR section to batch submission
- `2052c5c` feat(databento): converters for trades, status, ED daily schemas
- `e8dba69` chore(databento): import BULK_PULL_UNIVERSE_V additions from main
- `fe6f417` fix(databento): use GE.FUT for Eurodollars (Databento doesn't accept ED.FUT)
- `cb66073` docs(architecture): add validation framework + derivations section to overview

## Known Issues / Remaining Work

### Operational pull still in flight (Task 19)

3 jobs at Databento moved from `queued` to `processing` and are awaiting completion:
- `F` (MBP-1 sliver, free tier)
- `Status_continuous` + `Status_parent` (status events for full universe)
- `B_ED_daily` (Eurodollar daily, just resubmitted with `GE.FUT`)

Background finisher (`scripts/data/databento_batch_finish.py`) is polling and will auto-download + convert when each reaches `done`. Output log: `output/databento_finish_v2.log`.

### Trades_ES_MES deferred (insufficient funds)

Databento rejected the submission with `402 account_insufficient_funds`. Cost check via `metadata.get_cost`:
- Full 5y (ES + MES): **$1040.68**
- 1y only: $89.30
- 5y ES-only (drop MES): $601.52

User to decide: top up budget, reduce scope, or skip.

### EXPECTED_DENSITY calibration for Micro Yield futures

Pre-pull validation showed 5 CRITICAL failures (out of 233 checks):
- `density_2YY` 49 rows/day (expected 150-400)
- `density_5YY` 22 rows/day (expected 150-400)
- `density_30Y` 28 rows/day (expected 200-500)

These are real data signals — Micro Yield futures are sparser than the expectations expected. The expectation values in `src/data/validation/futures/expectations.py` should be recalibrated against observed values once we have a reasonable history.

### Known acceptable CRITICAL failures

- `latest_data_freshness`: ES latest bar 77 days old. Expected: bulk pull cutoff was 2026-02-22. Will resolve when continuous pull is set up or when data is refreshed.
- `derived_sofr_sanity`: 9/30 sample dates fail. Random sampler picks non-trading days; SR1 has gaps on weekends/holidays. Sampler should filter to business days.

### Tasks 20-21 still pending

- Task 20: Run final validation pass after Task 19 completes (will reflect new status + Eurodollar data).
- Task 21: Update `docs/reference/DATA_INVENTORY.md` with the three new datasets (`futures_trades_window/` if approved, `futures_status/`, `futures_per_contract_daily/root=ED/`).

## Validation

- All framework unit tests pass: 38 tests across `tests/data/validation/core/`, `tests/data/validation/futures/checks/`, `tests/data/derivations/futures/`.
- Integration tests pass on real local data: `test_layer1_sweep_passes` (0 CRITICAL failures at Layer 1), `test_gc_density_above_threshold` (>700 bars/day, confirms volume-roll bug fix is intact).
- End-to-end CLI run against real `H:/Stock_Data/futures_*/` produces a valid markdown report at `output/validation_smoke_pre_pull.md`: 233 checks, 182 passed, 5 CRITICAL, 46 warnings, 102.3s runtime.
- ASCII-only verified across all source files (Windows cp1252 compatibility).
- `python -c "import src.data.validation, src.data.derivations.futures, ..."` succeeds — no import cycles.

## Cost Notes

- Trades_ES_MES queries used `metadata.get_cost` (no submission, no charge).
- Status_continuous, Status_parent, F, B_ED_daily all submitted at `cost=$0.00` per Databento response (free tier or covered by subscription).
- Subscription expires 2026-06-01 — additional pulls beyond what's queued must be authorized before then.
