# FX Wave 3 Slate Pre-Registration - 2026-07-26

## Summary

Generated the FX Wave 3 candidate slate blind, per
`docs/strategies/research/20260726_fx_generation_brief.md`: 49 specs (39 runnable
catalog slots + 10 novel) across 10 mechanism families, each with mechanism,
fixed-parameter rule, computed viability screen, falsifier, spurious reason and
kill conditions. Two specs clear the bar. Along the way found and fixed a defect
in the blind-safe ledger builder that was presenting a tested-and-failed catalog
slot to the generator as OPEN.

## Changes Made

- **`scripts/strategy/build_generation_ledger.py`**: `parse_rows` used
  `_GRADE.get(cells[6], "OPEN")`, silently defaulting any unrecognized gate grade
  to OPEN. The tracker holds `'FAIL (cost-robust)'` (spaced); the map had
  `'FAIL(cost-robust)'` (unspaced). Catalog slot #20, tested and failed, was
  therefore shown to a blind generator as an open slot. Now raises on unknown
  grades; spaced variant added to the map.
- **`docs/strategies/research/20260726_fx_generation_ledger.md`**: rebuilt.
  OPEN 43 -> 42, TESTED 13 -> 14, READY-open 7 -> 6. SR_zero unchanged at 1.1807
  (N comes from the experiment registry, not the tracker).
- **`tests/strategy/test_build_generation_ledger.py`** (new): covers the grade
  map, the raise on unknown grades, both cost-robust spellings, and the
  deliberate WEAK -> TESTED-FAIL collapse.
- **`docs/strategies/research/20260726_fx_wave3_slate_prereg.md`** (new): the
  pre-registration itself, 49 specs.
- **`scripts/strategy/fx_wave3_slate_defs.py`** + **`build_fx_wave3_slate.py`**
  (new): the slate as executable data plus its renderer, so every published
  screen number is computed rather than hand-typed and the document is
  reproducible with `PYTHONPATH=$(pwd) python scripts/strategy/build_fx_wave3_slate.py`.

## Commits

- `6d04e06` fix(fx): ledger silently re-opened a tested slot as OPEN
- `73228f7` docs(fx): Wave 3 slate pre-registration -- 49 specs, 2 clear the bar

## Decisions and context not captured in the code

- **Blindness held, with one disclosed leak.** No forbidden file was read. The
  session-start git log in the system prompt contained the commit subject for the
  #20 re-gate verdict, which arrived before the brief was read. Slot #20 is
  excluded. The corrected ledger marks it TESTED-FAIL independently, so the
  exclusion does not rest on the leaked information.
- **Measurement split, deliberately.** `per_trade_vol_bps` was measured from the
  held 1m data (2011-2026) so denominators are honest; `gross_edge_bps` came from
  literature or first principles. Only UNSIGNED dispersion was measured. Signed
  drift, autocorrelation and continuation were deliberately not computed, because
  measuring the effect before proposing it conditions the pre-registration on the
  answer.
- **Two gates, not one.** A spec earns a trial only if it clears SR_zero AND
  still clears at 1.5x measured cost. This is what separates the slate: six
  high-trade-count intraday specs clear at measured cost (1.24-2.36) and collapse
  at 1.5x (0.31-1.03), i.e. they test the cost model more than the signal.
- **Self-audit moved the count from 19 to 2.** A first draft had 19 specs
  clearing. Auditing my own inputs found a trigger threshold used as an expected
  edge (spec 35), an edge contradicting its own stated prose (spec 14), and 12
  trade counts set to the maximum possible rather than the expected trigger rate.
  All corrected downward before publishing. Recorded in the document itself.
- **One prose error found by verification.** Spec 17 originally claimed the
  Sunday reopening has no measured quotes and takes the widest-hour fallback.
  Checking `hour_of_week_multiplier` showed those hours ARE measured (EURUSD
  3.84x at Sunday 22:00 UTC); the claim was corrected.

## Known Issues / Remaining Work

- **`screen_spec` has no concept of legs.** It charges one round trip regardless,
  flattering every relative-value spec. Each multi-leg spec in the document
  reports a leg-adjusted figure as authoritative, but the fix belongs in
  `src/backtesting/validation/viability.py` as an `n_legs` parameter.
- **`fx_round_trip_bps_at` ignores `_DERIVED_RT_BPS`.** EURGBP and GBPJPY have a
  derived-cross table in `costs/fx.py`, but the screen path looks up
  `_MEASURED_RT_BPS` only and falls back to the flat 4.0. Spec 35 inherits this.
  Worth reconciling, since the #20 re-gate showed the blanket fallback is not
  harmless.
- **The bar should be recomputed before any verdict.** SR_zero=1.1807 is quoted
  at N+50 as the ledger specifies. If only the 2 cleared specs are run, the true
  N is far lower. Recomputing it downward AFTER seeing which specs passed would be
  gate-tuning; fix N from pre-registered intent.
- **Nothing has been run.** The slate is pre-registered only. Execution of the 2
  cleared specs must go through `strategy-lead`, not this session.
- ML catalog slots 48-53 and DATA slot 55 remain unfilled by design: the
  triple-barrier meta-label harness does not exist and the PBOC fix data is not
  held.

## Validation

- `pytest tests/strategy/test_build_generation_ledger.py
  tests/backtesting/validation/test_viability.py -q` -> 17 passed.
- Ledger rebuild is idempotent and the diff was inspected: only the #20 row and
  the three affected counts changed.
- Renderer asserts no duplicate catalog slot, that all 39 runnable slots are
  covered, and that no slot outside the runnable set is proposed. All pass.
- Document verified ASCII-only (0 non-ASCII characters), 1763 lines.
