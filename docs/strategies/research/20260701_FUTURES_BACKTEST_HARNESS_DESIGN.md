# Futures Backtest Harness — Design Spec

**Date:** 2026-07-01
**Status:** Approved design, ready for implementation planning
**Context:** Closes Gaps A/B/C from `docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md`
(the futures backtest EXECUTION layer). The data/signal layer — continuous contracts, roll
calendar, carry, OI, contract specs — is already built and validated (roll calendar shipped
2026-07-01, cache built for all 53 roots). This spec builds the machinery that turns a futures
strategy's signals into a methodology-compliant backtest, proven end-to-end with one strategy.

---

## 1. Problem

Homeguard can compute futures signals (carry, roll dates, continuous-bar returns/vol, OI) but
CANNOT run a futures strategy backtest. Verified by audit 2026-07-01:
- `StreamingDataLoader` is equities/crypto only — no futures path (Gap A).
- `src/backtesting/costs/futures.py` covers 9 of 53 roots and is imported by nothing (Gap B).
- No futures portfolio simulator exists; the equity `PortfolioSimulator` is percent-cost +
  cash-mark-to-market (Numba-JIT) — the wrong basis for futures (Gap C).
- No futures strategy implementations, no futures backtest config, no runner path.
- `FuturesPositionSizer` is a 46-line stub used by nothing.

Result: you can do lightweight signal exploration in a scratch script, but you cannot produce a
net-of-cost equity curve, drawdowns, portfolio accounting, or anything that passes the
methodology's statistical gates.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | **Dedicated futures path** (new simulator + runner) | Correct futures semantics; zero risk to the live equity/crypto path OMR/RAMP/CSCM depend on |
| First deliverable | **Harness + Carver multi-speed TSMOM** (daily, multi-instrument) | Forces the full portfolio path; parameter-free strategy = no overfit surface; a real first result |
| Margin | **SPAN-style approximation**, replaceable module | True CME SPAN needs historical SPAN parameter files we don't have + a large risk-array engine; approximation gives ~80% of the effect and is accurate for outrights |
| Interface | **Config-driven YAML**, registry-integrated | Matches CLAUDE.md's config-driven rule; reproducible, registry-logged runs; strategy-lead pipeline compatible |

## 3. Scope

**In scope:** the daily multi-instrument futures backtest harness (Gaps A/B/C wired into a
runnable config-driven path) + Carver multi-speed TSMOM as the end-to-end proof + its config +
walk-forward + statistical-gate + registry integration.

**Out of scope (future specs):** true SPAN-file margin; intraday resolution; per-contract
carry/spread strategies; additional strategies beyond Carver; live-trading execution. Each real
strategy after Carver is its own spec -> plan -> build cycle on top of this harness.

## 4. Architecture & module map

Reuses Homeguard's asset-agnostic machinery; adds only the futures-specific pieces.

**New / changed modules:**
```
src/backtesting/
  engine/futures_portfolio_simulator.py   # NEW: daily multi-instrument sim (per-contract MTM,
                                           #   cost + margin integration) -> equity curve + trade log
  margin/futures_margin.py                 # NEW: MarginModel (scan-range margin + inter-commodity
                                           #   offset matrix + BP cap + utilization), replaceable
  costs/futures.py            (EXTEND)     # 9 -> 53 roots (from contract_specs); wired into sim
  utils/position_sizer_futures.py (FLESH)  # vol-target -> integer contracts, margin-capped
  data/futures_backtest_loader.py          # NEW: daily basket panel via ContinuousContractDataLoader
                                           #   (Gap A, dedicated — NO StreamingDataLoader change)
src/strategies/advanced/
  carver_momentum_strategy.py              # NEW: Carver multi-speed TSMOM (MultiSymbolStrategy)
  carver_indicators.py                     # NEW: 3 EWMAC pairs + forecast scalars
config/backtesting/carver_tsmom.yaml       # NEW: futures backtest config
src/backtest_runner.py      (EXTEND)       # route asset_class: futures -> futures path
src/data/futures/contract_specs.py (EXTEND)# add initial_margin / maintenance_margin fields
```

**Reused as-is (asset-agnostic — do NOT rebuild):** `WalkForwardValidator`
(`src/backtesting/chunking/walk_forward.py`), `StandardReportGenerator`
(`src/backtesting/reporting/standard_report.py`), the statistical gate (PSR/DSR/PBO, methodology
§2 — locate the existing functions; add to `src/backtesting/` only if genuinely absent), the
experiment registry (`append_run`, methodology §9.3), `ContinuousContractDataLoader`,
`RollCalendar`, `CarryCalculator`, `contract_specs`, `src/features/` primitives (EWMA, realized vol).

**Boundary discipline:** the futures path NEVER touches the equity/crypto `PortfolioSimulator`.
It is a separate class with its own daily loop. Zero risk to OMR/RAMP/CSCM.

**`contract_specs` extension:** add `initial_margin` and `maintenance_margin` (per-contract
scan-range $), keeping the single-source-of-truth pattern; extend the existing arithmetic
invariant test to cover the new fields where a relationship exists (e.g. maintenance <= initial).

## 5. Data flow

```
carver_tsmom.yaml
  -> backtest_runner (asset_class: futures)
  -> FuturesBacktestLoader: daily basket panel (ContinuousContractDataLoader.aggregate_to_daily,
                            ratio_adjusted) for the ~12-root basket
  -> CarverMomentumStrategy.generate_multi_signals(): per-instrument forecast in [-20,+20]
  -> FuturesPositionSizer: forecast + vol-target -> integer target contracts per instrument
  -> MarginModel: aggregate BP check; pro-rata scale-down if over cap
  -> FuturesPortfolioSimulator: daily loop, MTM P&L, costs on rebalance -> equity curve + trade log
  -> StandardReportGenerator + statistical gate + WalkForwardValidator
  -> experiment registry append_run + report .md/.json
```

## 6. FuturesPortfolioSimulator — the core new piece

Daily loop (the semantics that make this futures, not equities):

```
for each trading day d:
    # 1. Mark-to-market P&L on existing positions -> credited to cash daily
    daily_pnl = sum_i  contracts_i * multiplier_i * (close_i[d] - close_i[d-1])
    cash += daily_pnl

    # 2. On a rebalance day: target contracts (signals+sizer), diff vs current
    if rebalance(d):
        trades = target_contracts - current_contracts
        cost   = sum_i  abs(trades_i) * round_turn_cost(spec_i, hour=RTH)   # per-contract $, NOT %
        cash  -= cost
        current_contracts = target_contracts

    # 3. Margin: required = MarginModel.requirement(current_contracts)
    #    record utilization = required / equity; flag if > cap
    equity[d] = cash            # futures: equity IS cash (positions MTM'd into it daily)
```

**Correctness points:**
- **Return basis = ratio-adjusted continuous close.** The `.v.0` volume-roll already removes the
  roll discontinuity, so `close[d]-close[d-1]` is a clean return — NO separate roll-P&L term for
  the continuous path. (Verified during the roll-calendar work.)
- **Cost = per-contract dollars on contracts TRADED** (position diff), charged only on rebalance.
  Never a percent of notional.
- **Vol-targeting uses the same 25-day realized-vol basis as the Carver signal**, so sizing and
  signal are consistent.
- **1.5x cost-sensitivity gate** (methodology §4) = re-simulate with costs x1.5. Trivial since
  costs are a parameter.
- The simulator's ONLY output is a correct daily equity curve + trade log; all Sharpe/DD/CAGR/
  monthly math is the existing `StandardReportGenerator` (asset-agnostic, takes a pandas Series).

## 7. Carver TSMOM strategy + config

**`CarverMomentumStrategy`** (subclass `MultiSymbolStrategy`, mirrors CSCM):
- Basket: `[MES, MNQ, M2K, MYM, MCL, MNG, MGC, SIL, 6E, 6J, ZN, ZC]` (configurable).
- Per instrument: three EWMAC forecasts at the canonical Carver speeds `(4,16) (16,64) (64,256)`,
  each normalized by price-vol, scaled by Carver's forecast-scalar constants (`Systematic Trading`
  Table 19, hard-coded per pair), capped at +/-20. Combine equal-weight, cap aggregate at +/-20.
- **Parameter-free by design** — speeds + cap are DOCTRINE, NOT tunable. This is the reason Carver
  is the first strategy: no overfit surface, so the walk-forward is a clean read of the
  methodology, not a parameter search. NEVER grid-search these.
- Uses `src/features/` primitives (EWMA, realized vol) — no inline re-implementation
  (strategy-pipeline canonical-primitive rule).

**`carver_tsmom.yaml`:**
```yaml
asset_class: futures            # routes backtest_runner to the futures path
mode: multi_instrument
strategy:
  name: CarverMomentumStrategy
  parameters:
    universe: [MES, MNQ, M2K, MYM, MCL, MNG, MGC, SIL, 6E, 6J, ZN, ZC]
    speeds: [[4,16],[16,64],[64,256]]   # doctrine — fixed, documented as NOT swept
    forecast_cap: 20
    vol_target_per_instrument: 0.20     # annualized, pre-diversification
    rebalance: weekly                    # signal weekly, positions monthly (plan §4.1)
dates: {start: "2010-06-07", end: "2025-02-01"}   # reserve 2025-02+ untouched
backtest:
  initial_capital: 25000
  cost_model: futures_per_contract      # the extended 53-root model
  margin_model: span_approx             # scan-range + offsets
  data: {source: continuous, adjustment: ratio_adjusted, resolution: daily}
walk_forward: {train_months: 36, test_months: 12, step_months: 12}
output: {registry: true, report: true}   # append_run + .md/.json
```
The `asset_class: futures` key is the router; existing equity/crypto configs (no such key) are
untouched.

## 8. MarginModel (SPAN-style approximation)

One focused, replaceable module:
- **Scan-range margin:** per-contract `initial_margin`/`maintenance_margin` from `contract_specs`;
  portfolio requirement = sum |contracts_i| * margin_i.
- **Inter-commodity offset matrix:** small static dict of pairs -> credit % (ES/NQ ~75%, ZN/ZB
  ~70%; crack/crush later). Applied when both legs held opposite-signed. For TSMOM (outrights,
  same-direction basket) offsets rarely trigger — correct; the matrix is ready for spread strategies.
- **Buying-power cap + utilization:** aggregate requirement vs equity; sizer scales the book
  pro-rata if targets exceed the cap; daily `margin_utilization` recorded for the report.
- **Interface:** `requirement(positions) -> float`, `check_and_scale(targets, equity) -> targets`,
  so a real SPAN engine can replace it later without touching the simulator.

## 9. Position sizing

`FuturesPositionSizer` (flesh out the 46-line stub):
```
contracts_i = round( (forecast_i/10) * capital * vol_target * diversification_mult
                     / (multiplier_i * price_i * daily_vol_i) )
```
integer, hard-capped by `contract_specs.max_contracts`, then by the margin BP check.

## 10. Testing & acceptance

**Unit (TDD):**
- Simulator daily-MTM P&L on a hand-built 2-instrument scenario (known contracts x multiplier x
  price-diff = known P&L).
- Cost charged only on rebalance days; zero cost on hold days.
- Margin requirement with and without an offset pair.
- Sizer integer-rounding + margin pro-rata scale-down.
- Cost model returns correct per-contract $ for all 53 roots.

**Integration (real data):**
- Full Carver run producing a non-empty equity curve + trade log + registry `run_id`.
- 1.5x cost re-simulation.
- Walk-forward with 36/12/12 windows.

**Acceptance gates (methodology §2, §4, §5 — the real bar; use EXISTING gate functions):**
OOS Sharpe in a sane band; PSR/DSR with the project trial count; PBO; 1.5x cost-sensitivity still
above floor; IS/OOS degradation below threshold; per-regime robustness.

**Honest expectation:** Carver TSMOM realistic net Sharpe ~0.4-0.7 (plan §4.3). Deliverable
success = a **trustworthy, methodology-compliant result**, NOT a good Sharpe. A weak result is a
valid finding, not a harness failure.

## 11. Out of scope (future specs)

True SPAN-file margin; intraday resolution; per-contract carry/spread strategies; additional
strategies beyond Carver; live-trading execution.

## 12. Follow-up leads

- Extend the offset matrix + add true SPAN when spread strategies (crack/crush/curve) arrive.
- The other design-doc strategies (MOP TSMOM, carry, Donchian, seasonality, pairs) each become
  their own spec on this harness.
- If StreamingDataLoader ever needs a unified futures path, revisit; for now the dedicated
  `FuturesBacktestLoader` keeps the equity path untouched.
