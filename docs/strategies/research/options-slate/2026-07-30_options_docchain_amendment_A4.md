# Options Doc-Chain Amendment A4 — Hurdle Reconciliation and DSR Authority (2026-07-30)

**Date:** 2026-07-30
**Trigger:** A2 registered a hurdle computed by hand (1.02). The code path that will actually grade
Wave 1 computes a **different** number (1.137). A registered bar and an implemented bar that
disagree make a verdict unauditable. This reconciles them **before any Wave-1 execution**.
**Amends:** `2026-07-30_options_docchain_amendment_A2.md` §4 (the registered hurdle table)
**Instrument:** logged amendment, per the chain's registered rule. A2 stands unmodified; its §4
table is superseded here.
**Status:** still BLIND. **No strategy backtest has run. No P&L has been observed.**

---

## 1. The discrepancy

| Source | N | sigma_SR | E[max Sharpe \| null] |
|---|---:|---:|---:|
| **A2 §4 (registered by hand)** | 349 | 0.347 (theoretical, `1/sqrt(8.3y)`) | **1.02** |
| **`get_campaign_trial_distribution()` (what `gate_return_stream` actually calls)** | **141** | **0.4293** (empirical stdev of 130 observed trial Sharpes) | **1.137** |

They differ on **both** inputs and in **opposite directions** — the live path uses a *smaller* N but
a *larger* dispersion, and the dispersion dominates.

**Why N differs:** A2's 349 counts every evaluated specification, including individual sweep
points. The live function counts the static 40-trial SP-A/B/C/E baseline plus one Sharpe per
registry run carrying a numeric `oos_sharpe` (130 found). They are measuring different things, and
both are defensible for their own purpose.

---

## 2. Ruling — the live path governs, and the bar goes UP

**The registered Wave-1 hurdle is `E[max Sharpe | null] = 1.137`, as computed by
`get_campaign_trial_distribution()` + `src/backtesting/statistics/dsr.py` at evaluation time.**
A2 §4's table (1.02 / 1.024 / 1.036) is **superseded**.

Three reasons, in order of weight:

1. **It is internally consistent.** N and sigma_SR are derived from the **same empirical trial-Sharpe
   set**. A2's figure mixes a reconstructed count with a theoretical dispersion from a different
   source — a defensible hand-calculation, but not a coherent joint estimator.
2. **It is the conservative choice.** 1.137 > 1.02. When two defensible estimators disagree and no
   result has been observed, taking the harder bar is the only direction that cannot be accused of
   goalpost-moving later.
3. **It is what the code will actually do.** A registered number that the implementation does not
   produce is worse than useless — it invites a post-hoc "the real bar was X" argument.

**A2's N = 349 is NOT discarded.** It remains the honest record of lifetime search breadth and the
basis for `combinations_project`. It is simply not the gate input.

## 3. The bar is GROWING, and that is registered

`get_campaign_trial_distribution()` is designed to grow: each logged run appends a Sharpe, so N and
the hurdle rise monotonically as the search proceeds. This is deliberate — its docstring states the
intent as *"never shrinking N to make a gate easier."*

**Registered consequence: each Wave-1 candidate is graded against the hurdle computed at ITS OWN
evaluation time, not against 1.137 frozen at Wave-1 entry.** A candidate evaluated ninth faces a
higher bar than one evaluated first, because eight more trials have been logged by then.

This closes an otherwise real loophole — running the whole wave and then grading everything against
the entry-time N would under-count the search by exactly the size of the wave. **Every candidate
must report the N and hurdle actually used, not the entry-time figure.**

## 4. Authority — which function governs what (binding)

Three trial-count mechanisms now exist. Their roles are fixed here:

| Function | Value today | Authoritative for | NOT for |
|---|---:|---|---|
| **`get_campaign_trial_distribution()`** | N=141, sigma 0.4293 | **All strategy gates. The DSR input of record.** | ledger metadata |
| `n_trials_project_wide()` | 367 (post-fix) | `combinations_project` — **ledger metadata only** | **any gate** |
| Lifetime reconstruction (`20260728_lifetime_trial_count.md`) | 349 | the documented record of search breadth | any gate |

Before the pending DSR fix, `n_trials_project_wide()` returned 0 and was obviously broken. After
it, it returns a plausible-but-different number, which is **more** dangerous — a future reader
could mistake it for the gate input. **It must never be used to grade a strategy.** Unifying the
mechanisms is desirable future cleanup; it is not a Wave-1 blocker.

## 5. DSR applies to options, and N pools across asset classes

Recorded because both were questioned during this session:

- **DSR is applied to every options candidate.** It is not optional and is not weakened by any
  finding in this chain.
- **N pools across asset classes.** DSR's N measures *the researcher's multiple testing*. Having run
  hundreds of backtests across futures, FX, equities and options, reporting the best one carries
  that full multiplicity regardless of which class produced the winner. Segmenting N per asset class
  would grant a fresh cheap shot in every class touched — precisely the loophole DSR exists to
  close, and the reason the methodology mandates a **project-wide cumulative** count.

**The honest caveat, recorded rather than resolved:** Bailey-Lopez de Prado model N trials as draws
from a single null with dispersion sigma_SR. Our trials are not that — they span different strategy
families, windows and data frequencies, and the trial-Sharpe set is **futures-dominated**. The
theory is therefore being stretched by **either** estimator; neither 0.4293 nor 0.347 is the "true"
sigma_SR for an options candidate, because the setup does not cleanly define one. The conservative
stretch is taken deliberately. **This limitation must be stated in any Wave-1 results header, not
buried.**

## 6. Consequences

- Wave-1 candidates are graded against **≥ 1.137**, rising with each logged trial.
- **No candidate definition, parameter, prior, falsifier, wave assignment or kill switch changes.**
- A3's mark-source ruling for OPT-047/030 is unaffected.
- **Zero survivors becomes more likely still**, and remains pre-registered as a legitimate outcome.
  A 1.137 bar on a mechanism the feasibility screen calls *"compressed post-2022"* is a demanding
  test, and it is meant to be.

## 7. Honesty block

This amendment **raises** the bar it was written to reconcile. That is the direction that does not
require justifying — but it is not self-validating either, and two things should be checked by
anyone auditing it:

1. **The lower number was not chosen.** Both estimators were computed before any result existed;
   1.137 was taken over 1.02 on grounds of internal consistency and conservatism, and the reasoning
   is above rather than implied.
2. **The growing-N rule removes the most tempting future loophole** — grading a completed wave
   against its entry-time N. It is registered now, before it could be argued for after.

**What remains genuinely unsettled:** whether a futures-dominated trial-Sharpe dispersion is the
right sigma_SR for an options candidate. It is not, strictly. There is no better option available,
because **no options trial-Sharpe distribution exists yet — nothing has been tested.** After Wave 1
there will be nine, which is still too few to estimate a dispersion from. This is a known
approximation, taken in the conservative direction, and it does not become correct by being logged.

*End of Amendment A4. Ledger action: supersede A2 §4's hurdle table; append A4 rows to every Wave-1
candidate. Candidate definitions, priors, falsifiers and wave structure untouched.*
