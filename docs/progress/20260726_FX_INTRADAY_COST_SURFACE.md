# FX Intraday Cost Surface (Phase 0, item 1) - 2026-07-26

## Summary

Reviewed an external campaign post-mortem (`~/Downloads/20260726_fx_campaign_next_steps.md`),
verified its load-bearing claims against the repo, and found several materially
stale. Then executed the first item of the resulting Phase 0: replaced the
hour-blind intraday FX cost model with a MEASURED hour-of-week spread surface,
wired it into the only gated intraday strategy, and deleted the synthetic
artifact it supersedes. No verdicts were run.

## Verification of the external document

Checked before planning, because the document was written from a context that
predates this repo's last two sessions.

| Doc claim | Verified reality |
|---|---|
| Intraday engine is "the substantial build, not started" | Already built: `src/backtesting/engine/intraday_order_engine.py`, 4 test files, and one gated verdict (#20, OOS -1.60). The catalog tracker's own summary row contradicts its detail row 250 and is stale. |
| Need to download Dukascopy tick for a spread surface | Already owned locally: 14.1 GB raw tick bid/ask plus 0.93 GB minute-aggregated with spread percentiles, 5 majors, 2010-2026. |
| 1m dataset ~512 GB | 9.27 GB, 80 pairs, 2011-2026, fully materialised. |
| Cost model is "1.0 pip/side majors ~1.8 bp RT" | That model was deleted 2026-07-25 (`20b4dfa`). Measured EURUSD is 0.32 bps RT. |
| Open items: missing S&P leg, report clobbering | Both fixed in `ac20504`. |
| Open item: CB calendar has 2 dates | Half fixed. RBA/RBNZ backfilled 2011-2026, but the file header states the dates are rule-generated and must NOT be used for event-time studies. The doc's event-time specs need timestamps; we have approximate dates. |

Consequence for the doc's slate: every cost-viability number in its Section 4.3
was computed at the deleted 1.8 bps assumption, so its screen-kills are unsafe.
Its "session decomposition fails the cost screen at 1.8 bps RT" is revived at a
measured ~0.72 bps.

## The defect this session fixed

`src/data/artifacts/spread_model.py` was named an hour-of-week spread surface
and was not one. It read the real quote data, collapsed it to ONE median anchor
per pair, and emitted 168 identical rows plus a single 5x spike at hour 21 UTC.
It also had zero consumers. The live intraday cost path was instead
`fx_round_trip_pips`, the deprecated tier model, charging one constant per run.

Measured dispersion this was standing in for:

```
EURUSD  0.30 -> 10.20 pips across hour-of-week  (34x)
USDJPY  0.30 ->  3.80 pips                      (12.7x)
```

Error in the old flat model, round-trip bps, London hours 07-16 UTC:

| pair | flat model | measured London | ratio | measured weekend |
|---|---:|---:|---:|---:|
| EURUSD | 2.18 | 0.58 | 3.7x over | 3.05 |
| GBPUSD | 1.88 | 0.95 | 2.0x over | 4.45 |
| USDJPY | 2.00 | 0.69 | 2.9x over | 2.38 |
| USDCAD | 1.80 | 1.05 | 1.7x over | 8.65 |
| AUDUSD | 3.33 | 1.66 | 2.0x over | 18.48 |

Note the direction. Every apparatus defect found in this campaign so far has
been optimistic. This one over-charges the liquid hours and under-charges the
illiquid ones, so #20 London Breakout's gated OOS -1.60 is biased AGAINST
itself. It is still far from the ~1.13 bar, but the magnitude is not
trustworthy and the re-gate is owed.

## Changes Made

- **`config/costs/fx_hour_of_week_spread.csv`** (new, committed): 3016 rows, 25
  pairs. Stored as a MULTIPLIER on the per-pair level already in
  `_MEASURED_RT_BPS`, quote-weighted to mean 1.0, so the level keeps one source
  of truth and the pip-denominated sample contributes shape without a price
  conversion. Committed rather than read from local storage, matching the
  precedent set for the baked levels.
- **`scripts/data/build_fx_hour_of_week_cost.py`** (new): builds it. 5 majors
  from full-history local bid/ask; 20 further pairs from the existing Dukascopy
  sample. The hour-of-week dimension was already being computed by
  `measure_fx_spreads.py` and discarded at aggregation; this keeps it.
- **`src/backtesting/costs/fx.py`**: `load_hour_of_week_surface`,
  `hour_of_week_multiplier`, `fx_round_trip_bps_at`. Unquoted weekend hours
  charge the pair's WIDEST observed hour, not its mean. Unmeasured pairs get a
  flat 1.0, stated rather than invented.
- **`src/strategies/advanced/fx_london_breakout.py`**: charges half the round
  trip at the entry fill's hour and half at the exit fill's. `override_pips`
  kept as an explicit escape hatch to the legacy flat charge. New `cost_mult`.
- **`scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`**:
  `cost_mult` threaded through. This makes the methodology's mandatory 1.5x
  cost-stress leg buildable for the first time on this strategy; the runner
  previously documented that it could not rescale the summed daily R "without a
  strategy-level cost multiplier", which is exactly what was added.
- **Deleted** `src/data/artifacts/spread_model.py` and its test; dropped from
  the `fx_pipeline` registry (8 builders to 7, resolves cleanly).

## Phase 0 item 3: event calendar (the event-time gate)

The event-time strategy class was blocked on having real release instants. It
is now unblocked for US releases.

Measured error in the calendars that existed, over 2011-2026:

| calendar | rule | exact | mean error |
|---|---|---:|---:|
| `cpi.yaml` | 10th of each month | 14% | 3.93 days |
| `nfp.yaml` | first Friday | 83% | 1.15 days |

NFP's 24 bad months are off by exactly +7 days, i.e. the rule picked the wrong
Friday. Tolerable for a +-7d blackout, fatal for a T+2min entry, and silent.

Dates now come from the FRED releases API (BLS returns HTTP 403 to automated
fetches). Same-month duplicates are resolved STRUCTURALLY, never by market
outcome: selecting the bigger-moving date would bias exactly the event study the
calendar serves. CPI's February duplicate is the annual seasonal-adjustment
revision, 2 days before the main release in all 12 sample years.

Release times are verified against our own 1-minute data, not assumed. EURUSD
|return| in the release minute vs the same day's median minute, 2018-2023:

| event | events | release | background | ratio |
|---|---:|---:|---:|---:|
| NFP | 69 | 17.05 bps | 0.657 | 26x |
| CPI | 68 | 17.10 bps | 0.638 | 27x |
| FOMC | 47 | 15.13 bps | 0.636 | 24x |

The NFP profile confirms DST handling exactly: peak at 13:30 UTC in EST months,
12:30 UTC in EDT months, both 08:30 ET.

FOMC is scoped to 2013+ deliberately. Before that the statement alternated
between 12:30 ET on press-conference meetings and ~14:15 ET otherwise, with no
per-meeting flag available; the data shows that split, so one assumed time would
be wrong for about half of those 13 meetings. Left out rather than mis-stamped.

Still missing for full event coverage: non-US central banks (ECB/BoE/BoJ/BoC/SNB
have only a 2025-2026 starter set, and RBA/RBNZ are rule-generated dates that
their own config header marks as unusable for event-time work).

## Commits

- `d98eb35` feat(fx): measured hour-of-week spread surface for the intraday cost path
- `73e1ade` refactor(fx): delete the synthetic spread_model artifact, superseded by measurement
- `19c1488` feat(fx): charge #20 London Breakout the measured hour-of-week spread
- `f8f4b01` feat(fx): authoritative US macro release calendar with validated timestamps

## Validation

- New tests: 10 cost-surface tests, 4 strategy cost-wiring tests. All pass.
- One regression test is deliberate: `test_surface_is_not_flat_for_a_major`
  asserts EURUSD dispersion exceeds 5x, which the deleted synthetic model could
  never satisfy.
- Two existing London Breakout tests moved to the measured expectation. Their
  actual subject (the qty/pip-independent -1.0 R core) is unchanged and still
  asserted.
- Full `tests/strategies/` + `tests/backtesting/`: 1766 passed, 14 failed. All
  14 failures are futures spreads / futures register / rolling mode, confirmed
  pre-existing by re-running with the work stashed.
- Environment note: `tests/data/` has 6 collection errors from missing optional
  deps (boto3, databento, dukascopy_python, pandas_datareader) and 10 further
  pre-existing failures in futures/rates/sentiment. None related to this work.

## Known Issues / Remaining Work

1. **#20 re-gate is owed and is NOT done here.** It is verdict-producing work
   and belongs to `strategy-lead` behind the enforcement hook. This is the
   first defect that could move a number in the strategy's favour.
2. **New defect found, not fixed:** `src/strategies/advanced/fx_coint_scanner.py:89`
   charges `fx_round_trip_pips("major") * 0.0001`, a flat pip cost, on a DAILY
   multi-pair strategy that trades crosses. Same defect class as the daily cost
   model replaced on 2026-07-25, but on a path that computes its own cost
   instead of going through `fx_round_trip_usd`. Its verdict was FAIL (-0.24).
3. **Non-US central bank calendars remain unusable for event-time work.**
   ECB/BoE/BoJ/BoC/SNB have only a 2025-2026 starter set; RBA/RBNZ are
   rule-generated dates their own config header marks as approximate. The doc's
   CB-DRIFT spec spans 8 central banks and only FOMC is currently stamped.
4. Remaining Phase 0 close-outs: degenerate-signal tripwire, registry dedup
   guard, trial-count migration, exit schema in the intraday fills path.
5. Metals (XAUUSD/XAGUSD) show only 1.3-1.6x hour-of-week dispersion against
   10-19x for FX majors. Plausible for a broker-set metals spread, but it is
   unverified and worth a look before any metals intraday spec.
6. Stray untracked file in the repo root: `scripts.data.build_carry_cache`,
   apparently a mistyped shell redirect from an earlier session.
7. `scripts/backtest_scripts/` is gitignored while containing tracked files, so
   `git add` on a tracked runner there silently no-ops and needs `-f`. This
   nearly shipped an incomplete commit.
