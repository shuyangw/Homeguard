# Options Slate — Orchestration Session Log — 2026-07-27 → 2026-07-30

## Summary

Took delivery of an 8-document, externally-authored options pre-registration (50 candidates),
validated its data premises against disk, and ran Phase 0 → Phase 1 → Wave 0 → the first Phase-2
build. **No strategy backtest has been run and no P&L has been observed at any point.** The
headline outcome is that the chain's registered statistical bar was understated ~2.5x, and five
candidates were killed at zero trial cost before any of them consumed a draw from the sample.

## What was done

### Doc chain intake
Eight documents; three retired (the 2026-07-24 v1 set), five govern: Amendment A1, slate v1.1,
feasibility screen v2, handoff spec v2, and the Phase-0/1 work order. Committed to
`docs/strategies/research/options-slate/` as the pre-registration of record (`docs/research/` is
gitignored; only `docs/strategies/**` is tracked).

### Phase 0 — repo reconciliation (COMPLETE, 11/11, none ABSENT)
The existential item passed: `MarketRegimeDetector.analyze_regime_history()` is a **verified
causal replay** (truncates `index <= date`), and `_calculate_vix_percentile` reads a trailing
252-row window of an already-truncated frame — no full-sample lookahead. Regime gates are
backtestable; Wave 1 did not shrink.

Found instead: the options runner should follow the **FX/futures dedicated-runner pattern**, not
the equity registry; DuckDB is single-writer (constrains parallel sweeps); the existing options
cost model uses alpha = fraction of **half**-spread while the chain assumes **full** width (2x
trap); `OptionsDataLoader` strips the `_eod` suffix that marks a leak, and defaults to 16:00
rather than the registered 15:45.

### Phase 1 — canonicalization + V-battery (COMPLETE, 4,510/4,510 partitions, 24.08B rows)
Seven escalations. The load-bearing one: **ThetaData serves no usable IV/greeks for SPY/IWM/SPX
(every ETF/index root except QQQ) before 2017**, so the usable window is **~8.3 y, not 13.7**.

### Wave 0 — diagnostics (COMPLETE, zero trials consumed)
**5 killed** (OPT-040, 043, 042, 037, 005), 4 cleared, 2 blocked, 3 unresolved. Each died against
a threshold fixed before measurement, not because "the backtest looked bad."

### Lifetime trial count (COMPLETE)
Reconstructed **N ≈ 349** (range 179–621) against the chain's registered **9**.

### Amendment A2 — the registered hurdle
**0.41 → 1.02**, pre-committed before any Wave-1 execution and explicitly non-revisable downward.

### Phase 2 — M7 and the DSR fixes
M7 built (raw SVI, 270 partitions, 0 failures); the D-047/030 gate it was built to unblock
**FAILS**. Both DSR defects confirmed and fixed.

## Key decisions (logged)

| Decision | Resolution |
|---|---|
| ORATS (~$399) | **Deferred** → build M7 instead; missing-2008 caveat now permanent |
| ThetaData subscription | **Cancelled** (Phase 0.10 closed) → no refresh, no universe top-up |
| OPT-021 exit version | **v1.0 (T+1 open)**; v1.1 retired with its own ledger row |
| V12 legacy store | **Moot** — `options_1min/` does not exist on this machine |
| Trial-counting rule | **Every evaluated spec** — and see below, the alternative is not viable |
| Universe breadth | **Still open**; only "narrow" is free |

## Findings that outlive this slate

1. **`n_trials_project_wide()` has always returned 0** — filtered on `agent_name =
   'backtest-optimizer'`, which **no row has ever carried**. Anything on that path got zero
   deflation. The "optimizer-only" policy it encoded yields N=0 under *any* marker, because no
   optimizer sweep has ever written a row — so the choice was never stricter-vs-looser policy, it
   was **gate vs no gate**.
2. **`validation/deflated_sharpe.py` had a unit error** — subtracted a standard-normal quantile
   from an annualized Sharpe. 3.0x overstatement at N=349. Now a thin adapter to
   `statistics/dsr.py`; the wrong formula is deleted, not deprecated.
3. **`dsr == psr`, byte-identical, in 11 of 15 futures gate files** — i.e. **undeflated**. A
   larger corrupted population than both fixed bugs combined, already documented 2026-07-11 and
   still open. **Not an options problem.**
4. **RAMP Wave-3 DSR verdict is unreliable** — a trial chain was reset 36 → 1 and
   `RAMP_VARIANTS.md` records the family as passing at n_trials ≤ 12 and failing at ≥ 36. A
   verdict flipped on the reset.
5. **The NaN-vs-NULL trap** — greek columns are NaN-valued float64, not SQL NULL, so Arrow reports
   `null_count = 0` for a 100%-NaN column. Defeated three separate analyses including two of mine.
   Test `np.isnan()` on values; never `null_count`, never column presence.
6. **`RunStatus` had two distinct path-collision races** — the first fix made only the *tmp* path
   unique and left the destination shared at second resolution. Killed 9 of 276 build jobs before
   the second fix. Affects every long-running job in the repo.
7. **ThetaData's greeks boundary is documented** (v3 docs, not the marketing page): greeks are
   derived per tick from the underlying tick, and CTA-tape underlying history is limited. SPY is
   named explicitly. Not repairable by re-download.

## Errors I made, corrected in-record

- **Claimed the missing greeks were a recoverable download artifact.** Wrong — vendor boundary.
  I had checked *column presence* rather than whether values were real (the NaN trap).
- **Overstated the DSR blast radius** as "corrupts futures, FX and equities." Wrong —
  `combined_gate()` has zero production callers; that corpus runs through the correct
  implementation. The subagent refused the claim and checked.
- **Briefed a dividend approach that would have introduced lookahead** (realized yfinance
  dividends used to build an as-of-*t* forward). The M7 agent used put-call-parity forwards
  instead — point-in-time, self-validating (implied q 1.09–1.20% vs SPY's ~1.3%).
- **Over-applied `strategy-lead`** to the N reconstruction, which is accounting, not a verdict.

## Validation

Phase 0 by direct inspection with file:line evidence. Phase 1 over 4,510/4,510 partitions,
reconciled against on-disk counts. M7's D-047/030 threshold **committed before the first fit ran**
(`6271424`, docs-only, 132 lines) — verified independently. DSR test power demonstrated by running
the new suite against the restored old module: **11 of 14 failed**. Every headline number in this
log was re-verified in the main loop rather than taken from a subagent summary.

## Known issues / remaining work

- **OPT-047 and OPT-030 need a registered mark-source amendment** (raw quotes, not `iv_smooth`) —
  D-047/030 failed as registered. Raw 0.05-delta quotes exist in 100% of sessions, so the source
  is free, but the change must be logged before those candidates run. **This gates Wave 1.**
- **OPT-006 is blocked** — its 12–18M leg exceeds M7's registered 400-day cap. Needs an explicit
  amendment, deliberately not widened silently.
- **M7's far wing is the least trustworthy part of the surface** — p99 fit error 17.1 (SPY) / 23.7
  (QQQ) vol points below 0.05 delta, ~31% extrapolating. That is precisely where OPT-047 reads.
- **P1's low-delta rule rests on a premise that measurement contradicts**: shipped-greek delta
  error is *smallest* at low delta (median 0.0008) and *largest* deep ITM (0.0187) — the opposite
  end from the one the rule targets.
- Futures `dsr == psr` (finding 3) — open, and larger than anything fixed here.
- Universe breadth; the corporate-action adjustment layer (single names); earnings calendar
  (Wave 2); re-grading RAMP Wave-3.
- **Unpushed:** commits on `main` (some belonging to a concurrent session), plus branches
  `chore/lifetime-trial-count`, `feat/options-wave0-diagnostics`, `feat/m7-iv-surface`, and the
  DSR-fix worktree branch.

## Standing assessment

Wave 1 is 9 candidates against a pre-committed bar of **1.02**, on a mechanism the slate's own
feasibility screen calls *"compressed post-2022, with 12-month rolling VRP episodes turning
negative in 2023–24"*, over a window containing no crisis. **Zero survivors remains the single
most likely outcome and is pre-registered as legitimate.** Nothing measured so far argues against
proceeding; nothing measured so far argues that it will work.
