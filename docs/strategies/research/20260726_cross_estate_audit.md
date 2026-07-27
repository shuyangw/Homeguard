# Cross-Estate Apparatus Audit (P2.1) and Futures Fee Verification (P1.1)

**Date:** 2026-07-26
**Trials consumed:** 0. Everything here is code inspection, registry queries and
arithmetic. No backtest was run.

---

## Part 1: does the permissive apparatus reach beyond FX?

The premise being tested: the twelve FX defects were in *shared* code, and FX was
the one place permissiveness was harmless because there were no passes to
corrupt. The rest of the estate has passes.

**The premise is half right, and the half that is wrong is more interesting.**

### 1.1 Defect-class sweep across all 686 modules

| defect class | estate-wide status |
|---|---|
| `n_trials_project_wide()` used as a gate N (the SR_zero=0.0000 class) | **Clean.** One caller remains, `backtest_runner.py:680`, and it writes `combinations_project` as metadata, not a gate input. |
| `get_campaign_trial_distribution()` N-collapse under a registry lock | **Clean.** The defect was inside the function, now fixed; all 10 callers inherit the fix. |
| PSR/DSR unit mismatch (annualized Sharpe against per-period n) | **Clean.** AST audit of every `psr()`/`dsr()` call site across 686 modules: zero omit `periods_per_year`. |
| PBO stub-window truncation | **One dormant instance** -- see 1.2. |
| Same-bar fills / missing execution lag in the futures path | **Clean** -- see 1.3. |

### 1.2 The one live defect: `combined_gate` bypasses the PBO guard

`src/backtesting/validation/combined_gate.py:_pbo_via_splits_as_configs` truncates
every CPCV split column to the shortest split (`[:min_len]`) *without* first
dropping anomalously short splits. That is exactly the defect `_compute_pbo` was
fixed for on 2026-07-25: one short column drags every other column down to its
length, discarding most of the out-of-sample data before CSCV ever runs.

**Severity: low, because it is dormant.** `combined_gate` has no production
callers -- only `validation/__init__.py` (an export) and its own tests. Nothing
has been gated through it. But it is a shared gate module that a future
asset-class campaign would reach for first, so it should be fixed before it is
used, not after.

### 1.3 The futures path does not have the same-bar defect

`futures_portfolio_simulator._simulate` marks to market at step 1 and rebalances
at step 3, so a position established at date `d` earns the `d -> d+1` return, not
the `d-1 -> d` return that produced its own signal. Ordering verified by reading
the numbered steps. No lookahead.

### 1.4 The real finding: the live book was never gated at all

The audit expected to find live strategies gated under a permissive apparatus.
What the registry actually shows is different and, in one respect, worse.

| strategy | registry rows | gate statistics present |
|---|---:|---|
| RAMP (all versions) | 144 | `sharpe`, `pooled_oos_sharpe`, `psr_vs_0` |
| OMR | **0** | none -- absent from the registry entirely |
| CSCM | **0** | none -- absent from the registry entirely |

- **Zero explicit PASS verdicts exist in the registry**, across all 496 rows.
- **RAMP has no DSR and no PBO**, in any of its 144 rows. It was screened
  against PSR-vs-zero and never deflated for search, and never checked for
  backtest overfitting.
- **OMR and CSCM are not in the registry at all.** Two of the three live
  strategies have no machine-readable validation record.

So the correct statement is not "the live book was gated under a permissive
apparatus." It is: **two thirds of the live book was never entered into the
apparatus, and the third was gated on an incomplete subset of it.**

### 1.5 What RAMP looks like when deflated over its own search

RAMP's 144 registry rows are 144 configurations tried. That is selection, and no
deflation was applied to it. Deflating over its own logged search:

| quantity | value |
|---|---:|
| RAMP configurations logged | 144 |
| Observed Sharpe range | -0.550 to 0.910 (median 0.658) |
| Dispersion v | 0.2139 |
| **SR_zero over its own 144 configs** | **0.5681** |
| **Documented walk-forward OOS Sharpe** | **0.846** |

**RAMP clears its own deflated bar with a margin of +0.28.** That is a
reassuring result and it should be reported as such rather than dressed up as a
concern.

Three caveats, none of which change that conclusion but all of which bound it:

1. 144 is the *registry-logged* count. The true search was larger -- V26, V28,
   V31, V33, phases 3a/3b/4C -- so the real N is higher and the real bar
   therefore higher than 0.5681.
2. This is an SR_zero comparison, not a DSR. A DSR would also account for skew,
   kurtosis and sample length.
3. **PBO was never computed.** Nothing here speaks to overfitting.

### 1.6 Recommended actions, in priority order

| # | action | consumes trials |
|---|---|---|
| 1 | Enter OMR and CSCM into the registry with their validation vintage, or record explicitly that no machine-readable validation exists | 0 |
| 2 | Compute DSR and PBO for RAMP under the corrected apparatus (a re-gate of an already-counted trial, per the Kalman convention) | 0 |
| 3 | Fix `combined_gate`'s PBO truncation before any future campaign uses it | 0 |
| 4 | Make the `psr`/`dsr` caller audit estate-wide instead of a hard-coded list of 8 targets | 0 |

Action 2 is verdict-adjacent and belongs to `strategy-lead`. The consequence of
a failed re-gate must be written down **before** it runs, per the source
proposal, so it cannot be negotiated against a live P&L curve.

---

## Part 2: futures fee verification (P1.1)

### 2.1 Verified

**IBKR broker commission: USD 0.85 per contract per side** at the entry tier
(monthly volume <= 1,000 contracts), for the currency-futures block. Source:
`https://www.interactivebrokers.com/en/pricing/commissions-futures.php`,
retrieved 2026-07-26.

This confirms the repo's 6E broker component exactly, and it applies to the
whole currency block -- **not only 6E**. So the blanket `$2.50/side` estimate the
repo carries for the other seven FX roots is too high on the broker component.

### 2.2 Still unverified

**CME exchange and clearing fees.** `cmegroup.com` returns HTTP 403 to automated
fetches, as `bls.gov` and IBKR's other pricing pages did. The $1.00/side
exchange component for 6E is the repo's measured figure and is carried across to
the other roots below as **provisional**, not verified.

### 2.3 Corrected per-root cost table (provisional on the exchange component)

Broker $0.85 verified, exchange $1.00 provisional, so $1.85/side across all FX
roots:

| root | notional | repo now | corrected | all-in RT bps | was |
|---|---:|---:|---:|---:|---:|
| 6E | $135,000 | $1.85 | $1.85 | **0.737** | 0.737 |
| 6S | $141,250 | $2.50 | $1.85 | **1.147** | 1.239 |
| 6J | $83,750 | $2.50 | $1.85 | **1.188** | 1.343 |
| 6C | $73,000 | $2.50 | $1.85 | **1.192** | 1.370 |
| 6B | $79,375 | $2.50 | $1.85 | **1.254** | 1.417 |
| 6A | $66,000 | $2.50 | $1.85 | **2.076** | 2.273 |
| 6N | $60,000 | $2.50 | $1.85 | **2.283** | 2.500 |
| 6M | $29,000 | $2.50 | $1.85 | **3.000** | 3.448 |

6-major basket mean all-in round trip: **1.397 -> 1.266 bps**.

The "basket dominated by its worst legs" concern softens but does not vanish:
6A and 6N remain 2.8-3.1x the cost of 6E, driven by notional size rather than
fees, and 6M stays unusable at 3.0 bps.

### 2.4 P1.2 precondition FAILS

The source proposal requires confirming FX-root futures history is on hand
before locking a futures pre-registration. **It is not.** No 6E/6J/6B/6A/6C/6S
data exists in local storage, which holds only FX spot and options.

So an FX-futures regeneration cannot proceed as scoped. It needs a data
acquisition first (Databento GLBX.MDP3 or equivalent), which is a cost/benefit
decision, and per the proposal's own criteria P1.2 is "cancelled without
prejudice if FX-root futures data is not in inventory and its acquisition is
declined."

That decision has not been made, so P1.2 is **blocked, not cancelled**.
