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

## Phase 0 item 4: apparatus close-outs

**Degenerate-signal tripwire** (`src/backtesting/validation/degenerate_signal.py`).
Raises when a declared signal never varies. Wired into `compute_unwind_score`,
which retro-catches the exact EM seatbelt bug (all four terms fall back to a zero
series when their currency is absent, so on EM7 the filter was identically zero
and the strategy silently became plain carry), and onto the forecast panel in
`run_fx_backtest`. A DataFrame is degenerate only when EVERY column is constant;
one flat pair is a legitimate state and is logged instead.

**Registry duplicate detection.** Surfaces identical spec identities without
dropping them (duplicates inflate N, the safe direction, and N never shrinks).
The first version over-flagged and had to be corrected: grouping on NULL params
merged RAMP-V31's 47 rows, which carry 45 distinct metric sets and are plainly
different runs whose params were never recorded. Rows with NULL params are now
excluded.

Audit results, which contradict the post-mortem's assumption:

| finding | value |
|---|---|
| FX duplicate groups | **0** (the named RORO/PCA/Seatbelt cases do not reproduce) |
| All duplicate groups | 55, entirely futures, 167 extra rows |
| Test stubs in the PRODUCTION registry | 31 rows (ZeroForecastStub 16, ParamForecastStub 15) |
| Rows with no params at all | 150 of 496 (30%), spec identity unrecoverable |

**Trial-count migration.** The last two runners (FxCarrySeatbelt, #20 London
Breakout) sourced N from `n_trials_project_wide()`, which sums
`combinations_in_run` over `agent_name='backtest-optimizer'` rows. This campaign
logged none, so it returns **0**:

| | SR_zero |
|---|---:|
| what those two runners used | **0.0000** (the DSR gate reduced to "is the Sharpe positive") |
| correct campaign bar | **1.1382** (N = 141 + 1) |

Both also passed `[sharpe]`, a single-element list, as the trial-Sharpe
distribution instead of the campaign's 130 observed Sharpes; the dispersion was
degenerate. Both fixed together, since either alone leaves DSR meaningless. No
verdict is at risk (both strategies failed by wide margins) but any future run
through those runners faced no bar at all.

**Exit-side fill schema.** `Fill` now carries reason, trade_id, entry_ts,
entry_price, mae, mfe, bars_held. The engine was already computing the exit
reason and discarding it at the Fill boundary. Exits carry entry_ts/entry_price
so a round trip is reconstructable from the exit row alone; entries carry NaN
excursions rather than zero, because zero would be a claim.

## #20 re-gate (dispatched to strategy-lead)

Verdict: **FAIL, and now cost-invariant.** OOS Sharpe -1.87 at 1.0x, -2.71 at
1.5x. The zero-cost BOUND is the load-bearing number: with costs eliminated
entirely, OOS is still -0.14 and gross pre-cost P&L is +43.9R over 6162 trades
across 12 years. There is no gross edge, so the FAIL no longer depends on any
cost assumption.

My briefed premise was wrong on direction. I told the agent the cost fix runs
pessimistic so the number should improve from -1.60; it worsened, because
GBPJPY (the largest leg, 1888 entries) and EURGBP are in NEITHER measured table
and take the conservative 4.0 bps fallback. GBPJPY went 1.55 -> 4.40 bps and the
book average rose 1.30x. The agent declined the tempting fix (measure those two
pairs and re-run, a degree of freedom spent chasing a better number) and ran the
strictly stronger zero-cost bound instead, which makes the question moot.

Trial accounting: does NOT increment N (stays 141). Same pairs, params, window
and gate; no human chose a setting after seeing a result; the apparatus changes
were platform-wide and gate-TIGHTENING. Counting a bug fix as a trial penalises
fixing bugs. A flip to PASS was pre-committed to require counting it.

## Phase 2 prep

- **Blind-safe ledger** (`scripts/strategy/build_generation_ledger.py`). The
  tracker cannot be handed to a generator: its Notes column is full of OOS
  scores. The ledger exposes slot id, name, capability and a coarse status only.
  The WEAK grade is deliberately collapsed into TESTED-FAIL, because "this one
  nearly passed" is exactly what invites aiming at the near-miss.
- **Combination spec LOCKED** before any component exists, as a RULE not a list:
  every wave spec that cleared the viability screen and was run, equal weighted,
  static, one trial, and a component is NOT dropped for performing badly.
  Registered prediction: FAIL.
- **Generation brief** naming the files the fresh context must not read.
- Slot arithmetic: 43 OPEN + 4 naive-only, of which 40 are runnable (ML needs an
  unbuilt harness, DATA needs data we lack). A ~50 slate needs ~10 novel specs;
  the brief licenses a SHORTER slate over filler.

## Two defects in this session's own work, found by the re-gate

1. **I broke the runner.** `19c1488` made `bar` required on `_book_if_closed`
   and I updated the three call sites inside the strategy module while missing
   TWO in the runner. `py_compile` does not catch arity errors and the tests
   drive the strategy directly, never through the runner. The agent had to fix
   it before anything could run.
2. **`trade_id` was degenerate** -- 2 distinct values across 14838 fill rows.
   The counter was engine-local but the runner builds a fresh engine per day.
   My test passed by using ONE engine; the deployment uses one per day. Fixed
   with a module-level sequence plus a test that uses three separate engines.

Both are the same failure mode, and it is now the third occurrence this session:
a signature or state change that passes its unit test while being wrong in situ,
because the test does not reproduce the caller's topology.

## Governance note

I dispatched a git-writing subagent into the working tree I was committing in.
Its commits landed on `main` interleaved with mine (`8cbd84a`, then my
`a8292ac`, then its `fb169df`), and it reported them as being on a branch
`regate/fx-london-breakout` that does not exist, and attributed my commit to
"another session". Nothing was lost and both its commits are clean (no
settings.ini, no sentinel, verified), but I pushed its pre-registration
unreviewed as a side effect. My own stored memory warns against exactly this.
Use an isolated worktree for future agent-driven runs.

## Wave 3 resolution: zero runnable, zero trials

The slate (49 specs, generated blind in a separate session) resolved to NO
runnable specs, and no backtest was run. N stays 141, bar unmoved at 1.1807.

The viability screen routed 47 before any data was touched. The 2 survivors were
then killed by a cost term modelled the same day: IBKR's $2 per-order commission
minimum, confirmed from the account's own schedule. It stops binding above
$100k of notional per order; both survivors trade 6 majors concurrently, and the
account is cash-only (no ECP, so no leveraged spot FX). At $50k that is $8,333
per order, where commission alone exceeds the entire gross edge.

Cash spot would need $163,855 of capital for #18 and $113,931 for #21.

The pre-registered combination spec is VOID by its own K >= 3 rule, written
before any component existed so it could not be renegotiated afterwards.

SCOPE: this is an ACCESS and SIZE constraint, not a market finding. Nothing was
backtested, so it says nothing about whether the mechanisms are real. Stating it
as "month-end fix flow does not work" would be exactly the over-generalisation
the North Star warns against.

Live alternative: CME FX futures dissolve the constraint (one 6E is ~$135k
notional, above the minimum, ~0.185 bps/side). The repo already carries all 8 FX
futures contract specs, the cost model, the asset-class mapping, the futures
engine and a walk-forward runner. It is a different instrument, so it needs its
own pre-registration and its own trials.

## Commits

- `d98eb35` feat(fx): measured hour-of-week spread surface for the intraday cost path
- `73e1ade` refactor(fx): delete the synthetic spread_model artifact, superseded by measurement
- `19c1488` feat(fx): charge #20 London Breakout the measured hour-of-week spread
- `f8f4b01` feat(fx): authoritative US macro release calendar with validated timestamps
- `3c89460` feat(backtesting): degenerate-signal tripwire
- `fee7d85` feat(experiments): duplicate-spec detection for the registry
- `098d085` fix(fx): the last two runners deflated against a zero trial count
- `4ff0113` feat(backtesting): exit-side fill schema for the intraday engine
- `230ca1a` fix(backtesting): the trial count silently shrank when the registry was locked
- `8cbd84a` test(fx): pre-register the #20 re-gate (strategy-lead)
- `a8292ac` feat(backtesting): formal statistical-viability screen
- `fb169df` test(fx): #20 re-gate results -- FAIL, cost-robust (strategy-lead)
- `3e22171` fix(backtesting): trade_id collided across engines
- `d5b4386` docs(fx): Phase 2 prep -- ledger, combination spec, generation brief

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
4. **Registry hygiene, newly surfaced and NOT fixed:** 31 test-stub rows
   (ZeroForecastStub, ParamForecastStub) sit in the production registry, and 30%
   of all rows carry no params. Any project-wide N drawn from raw row counts
   includes the stubs. Deciding what to do with them is a trial-accounting
   question, not a code change, so it is left for review.
5. Metals (XAUUSD/XAGUSD) show only 1.3-1.6x hour-of-week dispersion against
   10-19x for FX majors. Plausible for a broker-set metals spread, but it is
   unverified and worth a look before any metals intraday spec.
6. Stray untracked file in the repo root: `scripts.data.build_carry_cache`,
   apparently a mistyped shell redirect from an earlier session.
7. `scripts/backtest_scripts/` is gitignored while containing tracked files, so
   `git add` on a tracked runner there silently no-ops and needs `-f`. This
   nearly shipped an incomplete commit.
