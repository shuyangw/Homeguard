# Futures Roll Calendar — Design Spec

**Date:** 2026-07-01
**Status:** Approved design, ready for implementation planning
**Context:** Closes "Gap D" from `docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md`.
The roll calendar is a prerequisite for all per-contract futures strategies (commodity
carry, FX carry, rates carry, inter-commodity spreads). Continuous-bar strategies (trend,
regime, mean-reversion) do NOT use it.

---

## 1. Problem

Per-contract futures strategies need to know, for any (root, date):
- which contract is the liquid "front" month
- which contract is "next" (for computing the term-structure / carry basis)
- when the front rolls from one contract to the next

We store raw per-contract bars (`futures/databento/per_contract_1min/`) keyed by raw CME
symbol (`GCG4`, `CLH4`, ...), but nothing tells us which contract is front on a given date.
Carry = `(F_front - F_next) / (F_next * days_between * 365)` is impossible without this.

This is the same class of problem that produced the original `.c.0` calendar-roll bug
(GC/CL at ~43 bars/day) that motivated the futures data rebuild. Getting the roll right
is what makes carry and spread backtests trustworthy.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Universe | **All 53 roots** | Future-proof; any later per-contract strategy just works |
| Roll signal | **OI-primary + volume tiebreak + calendar fallback, with FND clamp** | Matches Databento `.n.0` / vendor standard; OI empirically respects first-notice |
| "Next" contract | **Expose both**: by-cycle AND by-OI | Carry literature uses varying conventions; let backtest A/B them |
| Output | **Cached parquet per root** | Roll detection runs once; backtests do pure lookups |

## 3. Data sources (all confirmed on disk, 2026-07-01)

Data lives under the consolidated `H:/Stock_Data/futures/` tree (relocated by the
2026-04-20 consolidation; the older flat `futures_1min/` paths in the Phase-D plan doc are
stale and must be updated). Resolve via `from src.settings import get_local_storage_dir`.

| Purpose | Path | Verified |
|---|---|---|
| Contract specs / expiries | `futures/definitions/year=Y/month=M/data.parquet` | `raw_symbol`, `expiration`, `activation`, `cfi`, `unit_of_measure` present; 22 GC contracts clean |
| Open interest (primary signal) | `futures/databento/statistics/year=Y/month=M/data.parquet` | `stat_type=9` == open interest (28,400 rows/mo = 1/contract/day) |
| Volume (secondary signal) | `futures/databento/per_contract_1min/year=Y/month=M/data.parquet` | GCG4 65,541 vs GCJ4 1,997 on 2024-01-15 — front unambiguous |
| Volume-roll cross-check | `futures/databento/1min/` (`.v.0` continuous) | Trusted continuous series for validation |

### Data facts that constrain the design

1. **`contract_multiplier` in definitions is garbage** — `2147483647` (int overflow
   sentinel) for GC. Multipliers/tick-values MUST come from a static hand-verified spec
   table. `min_price_increment` (0.1 for GC) IS valid, so tick size can come from definitions.
2. **No first-notice-day field.** `expiration` = last-trade; `maturity_day` = `255`
   (Databento "not set" sentinel). FND must be derived from static per-family rules, not read.
3. **Settlement type is derivable** from CFI char-2 + `unit_of_measure`:
   physical = `FC*`/`FX*` + UoM in {TRYOZ, BBL, BU, ...}; financial(cash) = `FFI*` + IPNT (index).
   Deliverable-but-financial (FX `FFC`, bonds `FFD`) treated as physical for FND purposes.
4. **Per-contract data contains calendar-spread symbols** (`GCG4-GCJ4`, `CLN4-CLZ4`).
   These MUST be filtered out of roll signals — outrights only (no hyphen in `symbol`,
   cross-checked against `instrument_class='F'` vs `'S'` in definitions).
5. **Roots are not clean string prefixes.** `GC` vs `MGC`, `6E` vs option roots, etc.
   Match on the `asset` column in definitions, never string-prefix on `raw_symbol`.

## 4. Architecture

New package `src/data/futures/`, six focused units + a batch script + tests. Each unit
independently testable.

```
src/data/futures/
  contract_specs.py      # Static: root -> (multiplier, tick, tick_value, currency,
                         #   cycle_months, settlement_type, fnd_rule). Hand-verified vs CME.
                         #   Also consumed by the Gap-B cost model — build once, share.
  definitions_loader.py  # Reads futures/definitions/, returns per-root outright contract
                         #   list ordered by expiration. Filters instrument_class='F',
                         #   matches on `asset` column.
  roll_signals.py        # Extracts daily OI (stat_type=9) + daily volume per outright
                         #   contract. Excludes spread symbols. Pure reader, no roll logic.
  roll_calendar.py       # Core: roll-detection algorithm + RollCalendar lookup class.
                         #   Only stateful unit.
  __init__.py            # Public exports
scripts/data/build_roll_calendar.py   # Batch builder -> futures/roll_calendar/{root}.parquet
tests/data/test_roll_calendar.py      # TDD assertions vs known 2024 CME roll dates
```

**Cache artifact:** `futures/roll_calendar/{root}.parquet`, one row per date:
`(date, front_symbol, next_cycle_symbol, next_oi_symbol, dte_front, roll_trigger)`.
Backtests do pure lookups; roll detection runs once in the batch job.

**Separation rationale:** `contract_specs` is pure static data; `definitions_loader` and
`roll_signals` are pure readers with no decision logic; `roll_calendar` holds all roll
decisions and is the only stateful piece. Each testable in isolation.

## 5. Roll algorithm

Three-layer decision per (root, date):

1. **Classify once** (`contract_specs`): `settlement_type ∈ {financial, physical}` from
   CFI char-2 + `unit_of_measure`. Financial = no delivery risk. Physical = FND matters.

2. **Primary — OI crossover with hysteresis:** roll when back-month OI > front OI for
   N consecutive days (N tuned against golden dates, not doctrine; anti-whipsaw). Volume
   crossover breaks ties. Does most of the work for every root.

3. **FND safety clamp (physical roots only):** never hold a physical contract past a
   derived cutoff, even if OI data is missing/noisy. FND cannot be read, so encode a
   per-family offset rule (~6 rules, not 53 dates):
   - Metals (GC/SI/HG/PL): FND ≈ last business day of month before delivery month.
   - Energy (CL/HO/RB): last-trade already ~3d before the 25th; clamp = expiration − buffer.
   - Grains (ZC/ZS/ZW/ZL/ZM/KE): FND ≈ last business day before delivery month.
   - Rates deliverable (ZT/ZF/ZN/TN/ZB/UB): liquidity-driven roll well before delivery;
     clamp = expiration − buffer.
   - FX deliverable (6E/6J/...): clamp = expiration − buffer.
   - Financial cash (ES/NQ/M2K/MYM/SOFR/index/crypto): **NO clamp** — pure OI/expiry.

   The clamp only ever moves a roll EARLIER, never later. A wrong offset costs a little
   edge but can never create a delivery-risk artifact in a backtest.

**Key insight:** OI crossover already empirically respects FND (OI drains *because* traders
roll ahead of first notice), so the FND clamp is a safety net for missing/noisy data, not
the primary driver.

**Honest caveat (baked into spec):** FND offsets approximate true exchange rules. The
golden-date validation (Section 7) catches materially-wrong offsets.

**Calendar fallback:** where OI and volume are both missing (thin contracts, early history),
roll a fixed K business days before expiration. `roll_trigger` records which layer fired.

## 6. Public API

```python
@dataclass(frozen=True)
class ContractRef:
    raw_symbol: str        # "GCG4"
    expiration: date
    activation: date

@dataclass(frozen=True)
class RollEvent:
    roll_date: date
    from_symbol: str
    to_symbol: str
    trigger: Literal["oi_crossover", "fnd_clamp", "calendar_fallback"]

class RollCalendar:
    def __init__(self, cache_dir: Path | None = None): ...   # defaults to futures/roll_calendar/

    # primary lookups
    def get_front(self, root: str, on: date) -> ContractRef
    def get_nth_by_cycle(self, root: str, on: date, n: int) -> ContractRef  # n=1 -> next expiry in cycle
    def get_nth_by_oi(self, root: str, on: date, n: int) -> ContractRef     # n=1 -> 2nd most liquid

    # metadata
    def days_to_expiry(self, root: str, on: date) -> int
    def settlement_type(self, root: str) -> Literal["physical", "financial"]
    def roll_events(self, root: str) -> list[RollEvent]
```

**Fail-loud (per project rules):**
- Lookup for a (root, date) with no active contract raises `NoActiveContractError`, never
  returns `None`.
- Missing cache for a root raises at construction, not at first lookup.

Carry strategies call `get_front` + whichever `get_nth_*` their spec picks; the two `nth`
definitions are A/B-testable in a single backtest.

## 7. Testing & validation

TDD — assertions written before implementation, per project rules.

1. **Golden roll dates (acceptance gate):** hardcode known 2024 CME rolls for one root per
   family (GC, CL, ES, ZC, 6E, ZN) from published exchange calendars; assert the builder
   reproduces them within ±1–2 trading days.
2. **Cross-check vs `.v.0`:** assert OI-based front symbol matches the trusted volume-roll
   continuous series within a small window.
3. **Basis continuity:** assert carry `(front − next)/next` has no discontinuity spike
   across a roll (a jump = wrong roll date or contract mismatch).
4. **Spread-symbol exclusion:** assert `GCG4-GCJ4`-style spreads never appear as front/next.
5. **FND clamp:** assert physical roots (GC) roll before the derived FND cutoff; assert
   financial roots (ES) are unaffected by the clamp.
6. **Fail-loud paths:** `NoActiveContractError` on gap dates; construction error on missing cache.

**Killer validation:** GC and CL were the roots broken by the old `.c.0` calendar roll (the
43-bars/day bug). Reproducing published GC/CL 2024 roll dates is direct proof the original
problem is fixed.

## 8. Effort

~1.5 weeks single-developer. Delivers the shared `contract_specs` table that the Gap-B cost
model also needs.

| # | Task | Output | Est |
|---|---|---|---|
| 1 | Static contract-spec table (hand-verified vs CME) | `src/data/futures/contract_specs.py` | 1d (shared w/ Gap B) |
| 2 | Definitions reader → per-root contract sequence | `src/data/futures/definitions_loader.py` | 1d |
| 3 | OI + volume signal extractors (spread filtering) | `src/data/futures/roll_signals.py` | 1.5d |
| 4 | Roll-detection algorithm (OI + hysteresis + FND + fallback) | `src/data/futures/roll_calendar.py` | 1.5d |
| 5 | Batch builder → cache | `scripts/data/build_roll_calendar.py` | 0.5d |
| 6 | Tests vs known 2024 CME rolls | `tests/data/test_roll_calendar.py` | 1.5d |

## 9. Out of scope

- The Gap-A/B/C infra (futures loader, cost model, portfolio simulator) — separate specs.
- Any carry/spread strategy logic — this only provides the roll primitive they consume.
- Live-trading roll automation (IBKR `get_active_contract`) — backtest calendar only.
- Continuous-bar strategies — they use `.v.0`; applying this calendar to them is a
  double-roll bug.

## 10. Follow-up leads

- FND offsets are approximate; if golden-date tests show a family is off, refine that
  family's rule (not a 53-date hardcode).
- `stat_type` full code-table verification against Databento GLBX docs (we confirmed 9=OI;
  others assumed) — a one-day task if statistics-derived signals expand.
- Update `docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md` data paths to
  the consolidated `futures/` tree (stale flat paths throughout).
