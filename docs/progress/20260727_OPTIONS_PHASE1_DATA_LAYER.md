# Options Slate Phase 1 — Data Canonicalization + V-Battery — 2026-07-27

## Summary

Executed Phase 1 of the equity-options slate: built the canonicalization layer over the
233 GB `options_combined/` store and ran the registered V1-V13 verification battery. Five
escalation conditions fired, and one registered ruling's stated rationale was contradicted by
measurement. **No strategy backtest was run and no P&L was computed** — work order Section 4
prohibits it in this phase.

The headline is that the data differs materially from how the entire doc chain describes it.

## Changes Made

- **`src/data/options/canonical.py`** (new): `options_chain_1m` streaming read layer +
  `options_chain_eod` materialized at the registered 15:45 ET snapshot as a single guarded
  function. Preserves `_eod` suffixes (does not repeat the `data_loader.py:20-26` rename that
  strips the leak marker), normalizes `PUT/CALL -> P/C`, parses ET-naive ISO strings to
  `Datetime[us, UTC]`, implements the `>=1`-session `_eod` lag primitive.
- **`tests/data/test_options/test_canonical.py`** (new): 36 tests, all passing. Includes the
  leakage guard (no snapshot after 15:45) and the `_eod`-suffix regression test.
- **`scripts/data/build_vix_spot.py`, `src/data/acquisition/plugins/vix_spot.py`** (new):
  materializes VIX index spot with a provenance sidecar. Previously fetched at call time from
  yfinance `^VIX` — not reproducible.
- **`scripts/data/build_regime_state_daily.py`, `src/data/derivations/regime_state.py`** (new):
  materializes `regime_state_daily` once via the verified-causal `analyze_regime_history()`
  replay (it is O(n^2) with 8 downstream consumers). Causality regression test passes with a
  negative control.
- **`scripts/data/vbattery/`** (new): V4, V6, V7, V8, V10, V12, V13 measurement scripts, a
  schema/coverage inventory, and the V1/V2/V3/V5/V9/V11 sweep.
- **`docs/strategies/research/options-slate/20260727_options_vbattery_report.md`** (new): the
  deliverable — measured value, gate restated verbatim, PASS/PARTIAL/FAIL, evidence.

## Commits

- `7d122ee` feat(options): Phase 1a canonicalization + VIX/regime materialization + V-battery probes
- `8bd69e6` docs(options): Phase 1 V-battery report -- 5 escalations, V6 rationale corrected

Branch: `feat/options-phase1-data-layer` (not merged, not pushed).

## Escalations (require a decision before Phase 2)

1. **V7 — 7 of 7 corporate actions are AS-REPORTED.** Registered consequence: an adjustment
   layer is required before ANY single-name work. Non-standard deliverables confirmed present
   (TSLA 2022: all 120 post-split strikes off-grid at a 1.67 increment).
2. **V4 — an undocumented 16-column variant** (323 partitions, 17 roots, 2012-2016) carries no
   `implied_vol`/`delta`/`theta`/`vega`/`underlying_px`. V1's delta-bucketed gate is
   structurally unmeasurable there; V5 is 0% non-null there. Affects SPY, IWM, SPX and the
   ETFs — **not** the single names, and not QQQ.
3. **V8 — `root=META` 2021-07..2022-01 is Meta Materials, a different issuer** (median
   underlying 14.70 vs contemporaneous FB 361.88). Meta Platforms has a 7.4-month hole. A naive
   FB+META splice would insert a penny stock and bridge the gap silently.
4. **V8 — `root=SPX` contains no SPXW** (31/31 AM-settled third-Friday). Weekly/0DTE SPX work
   is not testable on this store.
5. **V9 — the store's edge is 2025-12-26, not 2026-02**, and SPY is truncated across 2017-H1.
   Combined with (2), the honest greek-bearing non-truncated SPY window is ~2017-09 -> 2025-12
   (~8.3 years), not the 13.7 years the chain assumes.

Additionally **V13**: no point-in-time membership source exists in the repo, so 4 of 5
registered universes are NOT MEASURABLE as registered; `U_INDEX` FAILS at 94.42% against the
95% gate. The gate was not adjusted to accommodate the miss.

## V6 — a registered ruling's rationale corrected

Phase 0 concluded that session *t*'s `open_interest_eod` is *t*'s end-of-day OI, making the
same-day join "a hard lookahead leak". That was inferred from the join being same-day; it was
never measured. Measured, three independent discriminators (volume/dOI correlation, the
`mean|dOI|/volume` ratio sitting at ~1.0 for the next-day alignment, and 85-95% of newly-listed
contracts showing OI=0 on their first traded session) say the stamp is **start-of-day OI
(= t-1's close)**. The same-day join is therefore not the leak it was described as.

`gamma_eod`'s date attribution remains **unresolved**.

**The `>=1`-session lag stands and is implemented.** It is conservative and safe under both
hypotheses, and the registered gate blocks OI/gamma work when the answer is not established.
What changed is the justification, not the rule. Relaxing it is the principal's call, not this
phase's — and it should not be relaxed for `gamma_eod` on current evidence.

## Known Issues / Remaining Work

- **The V1/V2/V3/V5/V9/V11 sweep is still running.** Definitions are validated against SPY
  2024-01 and reconcile with the known preliminary values (V1 [0.05,0.15] 99.986%, crossed
  0.00003%). Unswept root-years are reported as NOT MEASURED, never as PASS. The report must be
  updated when it completes.
- V8 SPX/VIX findings are windowed (last 12 and 24 of ~165 partitions); pre-2017 unmeasured.
- float32 precision loss in 274 partitions (2.18B rows) is recorded but its error magnitude is
  not quantified.
- `OptionsDataStore` deprecation (Phase 0.9) still awaiting go-ahead; nothing deleted.
- Corrected in passing: zero-bid frequency is **3.38%** full-month, not the 11.7% in the
  execution plan (that was a 614k-row sampling artifact).

## Validation

- 36 canonicalization + regime tests pass (`pytest tests/data/test_options/test_canonical.py
  tests/data/test_vix_spot_and_regime.py`).
- Snapshot leakage guard verified behaviourally on the materialized SPY 2024-01 EOD table:
  **0 of 68,615 rows** have a snapshot after 15:45 ET.
- `_eod` suffix preservation verified on the materialized output (no bare
  `open_interest`/`gamma`).
- V4 schema inventory independently reproduced by two implementations (4,510 partitions,
  24,078,079,007 rows) — and one bug was caught and fixed in each: a comma-split that
  mis-zipped dtypes containing commas, and a positional `column(0)` statistics read that is
  invalid because column order is not stable across partitions.
- META identity and SPX AM-settlement claims independently spot-checked before acceptance.

## Methodology notes

- Two subagent findings were rejected on first pass and re-verified before entering the report;
  one of my own measurements (a per-root truncation sweep) produced fabricated zeros from the
  positional-column bug and was discarded rather than reported.
- `TODO.md` is gitignored in this repo, so phase tracking there is local runtime state only.
