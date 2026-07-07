# Tier-1 EUR/GBP Event Calendar Design Spec

**Date:** 2026-07-07
**Status:** Approved (brainstorm), pending implementation plan
**Context:** Sub-project 2a of the intraday FX engine. The intraday vertical slice targets research strategy #20 (London Open Breakout), whose entry filter skips days with a tier-1 EUR/GBP economic release in the 09:30-12:00 London window. No historical intraday-timestamped release calendar exists (the current `config/macro_calendar/cb_decisions.yaml` is date-only and a 2025-2026 starter). This sub-project builds that calendar as a keyless, reusable asset consumed by the #20 slice (sub-project 2b) and later intraday strategies (#21-23). Depends on the merged FX session clock (`src/backtesting/sessions/fx_clock.py`).

## 1. Purpose

Provide a keyless, version-controlled calendar of tier-1 EUR and GBP macro releases with London-correct release times, plus a predicate that answers "does a tier-1 EUR/GBP release fall in [window] on [day]?" so intraday strategies can skip event-driven days. Built from curated recurring RULES (not scraped actuals), which is keyless and covers the full backtest range at the cost of approximate dates. Because consumers use it as a day-level skip filter (exclude event days, never trade them precisely), approximate dates are acceptable.

## 2. Constraints and inputs

- Keyless and pure: rules live in a version-controlled yaml; the module reads the yaml and expands rules deterministically. No network, no API key.
- Release times are placed on the UTC timeline via `src.backtesting.sessions.fx_clock.local_to_utc` (DST-correct). Reuses `config/macro_calendar/cb_decisions.yaml` (via the existing `src.data.macro_calendar.load_cb_decisions`) for ECB/BoE decision dates rather than re-listing them.
- ASCII-only, no em dashes, no emojis, no print(). Use `from src.utils import logger` only where genuinely needed.
- Follows the existing `src/data/macro_calendar.py` module pattern.

## 3. Architecture

Two files: a curated rules yaml and a generator/query module.

### 3.1 Rules yaml: `config/macro_calendar/eurgbp_tier1_releases.yaml`
A list of recurring release rules. Each rule:
- `name`: str, e.g. `EZ_FLASH_CPI`, `UK_CPI`.
- `currency`: `EUR` or `GBP`.
- `cadence`: one of a small fixed vocabulary the generator understands:
  - `monthly:nth-weekday:<n>:<weekday>` (e.g. `monthly:nth-weekday:3:WED` = third Wednesday each month)
  - `monthly:month-end-business-day` (last business day of the month)
  - `monthly:business-day:<n>` (nth business day of the month)
  - `quarterly:nth-weekday:<n>:<weekday>:<month-offset>` (quarterly, in months matching an offset)
- `time_local`: `"HH:MM"` and `tz`: an IANA zone (e.g. `Europe/Berlin` for EZ, `Europe/London` for UK) giving the typical release time.
- `overrides` (optional): explicit `date -> time_local` entries for known one-offs.
- `from_cb_decisions` (optional): a CB key (`ECB` or `BOE`) meaning "take dates from cb_decisions.yaml for this currency" with the given `time_local`/`tz`.

Header documents that dates are approximate recurring-rule expansions, not scraped actuals.

### 3.2 Module: `src/data/macro_calendar_tier1.py`
Functions:

```python
def load_tier1_rules() -> list[dict]
```
Read and return the rules from the yaml (raises FileNotFoundError with a clear message if absent).

```python
def generate_tier1_releases(start: datetime.date, end: datetime.date) -> pd.DataFrame
```
Expand every rule over `[start, end]` into rows `(date, name, currency, release_utc)` where `release_utc` is a tz-aware UTC `pd.Timestamp` from `fx_clock.local_to_utc(tz, combine(date, time_local))`. Rules with `from_cb_decisions` pull their dates from `load_cb_decisions()`. `overrides` replace the generated time for their dates. Sorted by `release_utc`.

```python
def tier1_release_in_window(day: datetime.date, win_start: datetime.time,
                            win_end: datetime.time, exchange: str = "LONDON",
                            currencies: tuple[str, ...] = ("EUR", "GBP"),
                            releases: pd.DataFrame | None = None) -> bool
```
True if any release for `currencies` on `day` has a `release_utc` whose exchange-local wall time falls in `[win_start, win_end)`. Converts `release_utc` to `exchange` local via the fx_clock exchange zone. Accepts an injected `releases` frame (for tests / to avoid regenerating); when None, generates for `[day, day]`.

### 3.3 Event set (initial rules)
- EUR: `EZ_FLASH_CPI` (month-end business day, 11:00 Europe/Berlin), `EZ_FLASH_GDP` (mid-month, 11:00 Europe/Berlin), `EZ_FLASH_PMI` (~3rd-from-last business day, 10:00 Europe/Berlin), `DE_IFO` (10:00 Europe/Berlin), `DE_ZEW` (11:00 Europe/Berlin), `DE_FLASH_CPI` (10:00 Europe/Berlin), `ECB_DECISION` (from_cb_decisions ECB, 13:45 Europe/Berlin).
- GBP: `UK_CPI` (monthly, 07:00 Europe/London), `UK_JOBS` (monthly, 07:00 Europe/London), `UK_GDP` (monthly, 07:00 Europe/London), `UK_RETAIL_SALES` (monthly, 07:00 Europe/London), `UK_PMI` (monthly, 09:30 Europe/London), `BOE_DECISION` (from_cb_decisions BOE, 12:00 Europe/London).

Exact cadence anchors (which weekday/business-day) are set to plausible typical values in the yaml and are tunable; correctness of the FILTER depends on the release TIME landing in/out of the window, which is structural per event, not on the exact day.

## 4. Behavior notes / correctness

- Window semantics: half-open `[win_start, win_end)`, matching fx_clock's session convention.
- The CET/CEST (Europe/Berlin) and London offset is a constant 1 hour year-round (EU-harmonized DST), so EZ releases map to a stable London wall time; no offset-divergence trap. fx_clock handles the conversion regardless.
- Correctly-encoded consequence: UK 07:00 London releases fall before #20's 09:30-12:00 window (no skip); EZ 10:00-11:00 Berlin (09:00-10:00 London) and UK PMI 09:30 / BoE 12:00 fall in or at the window edges and are evaluated by the predicate.

## 5. Known limitations (documented)

1. Dates are approximate recurring-rule expansions, not scraped historical actuals; a real release may shift a day or two. Acceptable because consumers use this as a day-level skip filter. Exact-actuals sourcing is a deferred future upgrade.
2. The event set is the tier-1 core, not exhaustive; adding a rule is a one-line yaml edit.
3. `cb_decisions.yaml` currently covers only 2025-2026; ECB/BoE decision coverage before 2025 is sparse until that file is backfilled (tracked separately, out of scope here).

## 6. Files

- Create `config/macro_calendar/eurgbp_tier1_releases.yaml`
- Create `src/data/macro_calendar_tier1.py`
- Create `tests/data/test_macro_calendar_tier1.py`

## 7. Testing plan

Tests in `tests/data/test_macro_calendar_tier1.py`:
1. `load_tier1_rules` returns the expected rule names for EUR and GBP.
2. `generate_tier1_releases` over a sample year yields rows with tz-aware UTC `release_utc`, sorted, covering both currencies.
3. An EZ flash-CPI release converts to a London wall time inside 09:30-12:00 -> `tier1_release_in_window(day, 09:30, 12:00)` is True.
4. A UK 07:00 London release is NOT in the 09:30-12:00 window -> predicate False on a day with only that release.
5. A UK PMI at 09:30 London is in the window (half-open lower edge) -> True.
6. DST-boundary release (a Berlin release near an EU DST change) still converts to the correct London wall time via fx_clock.
7. `from_cb_decisions` ECB/BoE dates appear in the generated frame with the configured time.
8. Injected `releases` frame path: `tier1_release_in_window(..., releases=df)` does not regenerate and honors the passed frame.

## 8. Out of scope (deferred)

- Scraped/exact historical release timestamps (this builds curated recurring rules).
- Release actual-vs-forecast surprise magnitudes (this is a schedule calendar only).
- US / JPY / other-currency tier-1 events (EUR/GBP only, for #20).
- The #20 strategy, intraday loader, and OCO engine (sub-project 2b).
- Backfilling `cb_decisions.yaml` pre-2025 (tracked separately).
