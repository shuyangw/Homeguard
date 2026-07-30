# Options Doc-Chain Amendment A2 — Trial-Count and Window Correction (2026-07-30)

**Date:** 2026-07-30
**Trigger:** Phase-1 V-battery measurement of the owned store (2026-07-27) + lifetime trial-count
reconstruction (2026-07-28). Both are **data and accounting facts**, not performance observations.
**Amends:** `2026-07-25_options_strategy_slate_v1_1.md` · `2026-07-25_options_slate_feasibility_screen_v2.md` ·
`2026-07-25_options_slate_cc_handoff_spec_v2.md` · `2026-07-25_options_docchain_amendment_A1.md`
**Instrument:** logged amendment, per the chain's own registered rule -- *"deviations require a
logged amendment before the affected test runs"*, and *"superseded versions are retired with their
own ledger rows, **never overwritten in place**."* The four amended documents stand unmodified in
the record; this document supersedes the specific sections enumerated in §3.
**Status:** still BLIND. **No strategy backtest from this slate has run. No P&L has been observed
by any stage of this chain.** Every change below is driven by measurement of data properties and
by reconstruction of the trial ledger. Neither is results-conditioning.

---

## TL;DR

- **The registered Wave-1 DSR hurdle of 0.41 is understated by ~2.5x. The honest bar is ~1.02.**
  Both inputs to it were wrong in the same direction and they compound: the usable window is
  **8.3 years, not 13.7**, and the trial count feeding DSR is **~349, not 9**.
- **The pre-registration's arithmetic was never the problem.** The reconstruction reproduces the
  chain's own table exactly (N=9 -> 0.4109 vs the doc's 0.41; N=19 -> 0.5074 vs 0.51; N=31 ->
  0.5638 vs 0.56). The multiplier was right; the **inputs** were wrong.
- **The N uncertainty does not change the decision.** N is genuinely uncertain across 179-621, but
  because the null max grows as ~sqrt(2 ln N), that entire 3.5x span moves the bar only **14%**
  (0.95 -> 1.08). Every defensible accounting rule lands near 1.0. **None lands near 0.41.**
- **Why the window shrank:** ThetaData serves no usable IV/greeks for SPY/IWM/SPX (and every
  ETF/index root except QQQ) **before 2017** -- a documented vendor coverage boundary, not a bad
  download. Greeks are derived per tick from the underlying tick, and the underlying history does
  not exist for CTA-tape symbols in that era.
- **Nothing here upgrades or downgrades any prior.** Candidate definitions, fixed parameters,
  families, priors, falsifiers, ablation ladders, wave structure and kill switches are all
  **unchanged**. What changes is the bar a survivor must clear, and the window it is measured on.

---

## 1. Why this is an amendment and not an edit

A1 established the procedure and it is followed here: conversation-chain documents are corrected
by a **new logged amendment**, not by editing, with each superseded claim enumerated and replaced.
The Phase-0/1 work order independently prohibits edits to the pre-registration documents
(`2026-07-25_options_phase01_cc_work_order.md` §4).

Nothing amended here has been tested. The amendment is registered **before** any Wave-1 execution.

---

## 2. The corrected inputs

### 2.1 Window: 13.7 y -> 8.3 y

| | Registered (A1 §8) | Corrected |
|---|---|---|
| Primary window | 2012-06 -> 2026-02 (13.7 y) | **2017 -> 2025 (~8.3 y)** for SPY-based candidates |
| sigma_SR = 1/sqrt(years) | 0.27 | **0.347** |

**Cause.** Greeks are **derived per tick**, not stored: *"Theta Data calculates Greeks for each
tick of data and uses the exact underlying tick (price) at the time of the option tick."* Underlying
coverage is **tape-dependent**: full UTP history from 2012-06-01, but *"for symbols only available
on the CTA tape, the history is limited to 2020-01-01"* -- **SPY named explicitly**. No underlying
tick history => no greeks.

Measured in our store: **3,600 of 4,510 partitions (79.8%) have usable greeks**; 580 are `ALL_NAN`,
323 `NO_COLUMN`, 7 `PARTIAL_NAN`. In every `ALL_NAN` partition `underlying_px` is **also** 100%
NaN, while `bid_close`/`ask_close` are 100% intact -- the documented mechanism, visible in the data.
Every ETF/index root **except QQQ** has first-fully-usable year **2017**. Per-partition census:
`20260728_options_greek_coverage_census.csv`.

Additionally, **SPY 2017 H1 is effectively absent** from the store (2017-01 holds 2 of ~20
sessions; 107 missing sessions across H1) -- a separate defect from the greeks boundary, and the
reason the practical window starts mid-2017.

**Not repairable by purchase from ThetaData** -- re-downloading returns the same empty columns.
Recoverable only for **2016** by self-computing IV from surviving quote mids plus our own 1m
underlying (which starts 2016), and only via ORATS for anything earlier.

### 2.2 Trial count: N = 9 -> N ~ 349

Reconstructed from `output/experiments.duckdb` (496 runs at reconstruction time) plus off-ledger
campaigns. Method and every judgment call: `20260728_lifetime_trial_count.md`.

| | Registered | Corrected |
|---|---|---|
| N feeding the options DSR | 9 | **349** (range **179-621**) |
| N_total (all-time) | -- | 367 |

The **overlapping-window rule buys almost nothing**: only 6 distinct specs are calendar-disjoint
from 2017-2025, because every campaign window (futures 2010-2026, FX 2011-2026, equity 2017-2026)
spans it. N_total 367 -> N_overlapping 349.

**Three findings that materially affect the count, recorded here:**

1. **`n_trials_project_wide()` has always returned 0.** It filters `WHERE agent_name =
   'backtest-optimizer'`; no registry row has ever carried that agent_name. Since
   `expected_max_sharpe()` returns 0.0 for N<2, **any DSR routed through it received zero
   deflation.** This is also why `combinations_project` was 0 everywhere. Fix in progress under a
   separate task; **until it lands, no DSR output from that path is trustworthy.**
2. **An equity trial chain was reset 36 -> 1.** `RAMP_VARIANTS.md` states the V20+ family "starts
   at `n_trials_project = 1` (fresh chain)" and -- decisively -- "passes at n_trials <= 12
   (Wave-3 reset), fails at >= 36." **A gate verdict flipped on the reset.** The 36 is re-added to
   lifetime N, and **the RAMP Wave-3 DSR verdict must be treated as unreliable pending re-grading.**
3. **The chain's "options n_trials = 18"** is one strategy's optimizer grid, not the campaign
   total. The repo records 31 candidates catalogued, **16 tested**. 16 is counted. (It also
   collides numerically with the OpEx strategy's unrelated "18 trades" -- **do not sum them**.)

---

## 3. Errata — superseded -> replacement

### 3.1 Amendment A1 §8 ("Revised DSR arithmetic")

| Superseded | Replacement |
|---|---|
| "Primary window 2012-06 -> 2026-02 ~ **13.7 y** => sigma_SR ~ 0.27" | **~8.3 y for SPY-based candidates => sigma_SR ~ 0.347.** QQQ retains ~13.7 y (the only index root with full usable greeks) |
| Null best-of-N table: Wave 1 (N=9) ~ **0.41**; +Wave 2 (N~19) ~ 0.51; +Waves 3-4 (N~31-38) ~ 0.56-0.59 | **Superseded by §4 below.** Those figures assumed N counts only this slate's trials. DSR's N is **project-wide and cumulative**, so the correct entry point is N ~ 349, not 9 |
| "missing-2008" caveat | **Retained and strengthened.** Now permanent absent an ORATS purchase, and it right-censors the loss distribution of the short-vol core |

### 3.2 Feasibility screen v2 §7 and handoff spec v2 §7.2

Both carry the same superseded table (N=9 -> 0.41 etc.) and the same 13.7 y / sigma 0.27
assumption. **Both superseded by §4.** The per-root-window rule they state (each candidate's DSR
uses **its own** window; mixed-window baskets report the binding shortest root) is **retained and
now binding**, since roots diverge much more than the chain assumed: QQQ ~13.7 y, SPY ~8.3 y,
PLTR ~5.7 y, COIN ~4.7 y.

### 3.3 Handoff spec v2 §6, "Universal pre-registered criteria"

> "report DSR against lifetime N"

**Retained verbatim, and now operative with an actual number.** Lifetime N is **349**, sourced from
`20260728_lifetime_trial_count.md`, **not** from `n_trials_project_wide()` until that function is
fixed, and **not** from the wave.

### 3.4 What is NOT amended

Candidate definitions and fixed parameters · family assignments · prior tiers · falsifiers ·
ablation ladders · wave structure and membership · kill switches and the prohibited-responses list ·
the diagnostics and their registered gates · the +50 emitted-trial delta · OPT-023's NO-GO ·
OPT-021's v1.0 exit decision · the reiterate criteria · the rule that STOP decisions on capital are
Shuyang's.

---

## 4. The registered hurdle (this replaces every prior table)

sigma_SR = 1/sqrt(years). Multiplier per Bailey-Lopez de Prado expected-max-Sharpe, computed via
`src/backtesting/statistics/dsr.py` (**not** `validation/deflated_sharpe.py`, which carries a unit
error -- see §5).

| Stage | N | sigma_SR (8.3 y) | **E[max Sharpe \| null]** |
|---|---|---|---|
| **Wave 1 entry** | **349** | 0.347 | **1.02** |
| Post-Wave-1 | 358 | 0.347 | **1.024** |
| Post-full-slate | 399 | 0.347 | **1.036** |
| *sensitivity: N low* | 179 | 0.347 | 0.95 |
| *sensitivity: N high* | 621 | 0.347 | 1.08 |

**Pre-committed now, before any result is seen:** Wave-1 candidates are graded against
**E[max | null] = 1.02**. Post-Wave-1 additions use **1.024**; a full-slate run uses **1.036**.
These are fixed at this document's date and **may not be revised downward after a result is
observed.**

**Per-root exception (registered):** a candidate whose universe is QQQ-only may use sigma_SR based
on its own longer window; it must say so explicitly in its results header, and any comparison
against a SPY-based sibling must be run on the **matched** window. The OPT-015 / OPT-019 ablation
in particular is bounded by the 1m-underlying history (2016+) regardless of option coverage.

---

## 5. DSR implementation — binding instruction

Two implementations exist and disagree by ~3x:

- **`src/backtesting/statistics/dsr.py` — USE THIS.** Verified to reproduce the chain's own
  arithmetic exactly.
- **`src/backtesting/validation/deflated_sharpe.py` — DO NOT USE pending fix.** It computes
  `expected_max_sr` as the expected maximum of a **standard normal** (~1.97 at N=18) and then
  subtracts it from an **annualized Sharpe** -- a unit error that makes DSR nearly unpassable.
  Live example in-repo: `docs/reports/ramp-long-calls/20260402_statistical_validation.json`
  records `expected_max_sharpe: 1.9733` at `n_trials: 18`, `p_value: 1.0`.

A consolidation task is in flight. **Until it lands and its regression tests pass, no DSR figure
from the `validation/` path may be quoted as a verdict** -- in this slate or any other.

---

## 6. Consequences

1. **The bar roughly doubles-to-triples.** A Wave-1 candidate must clear ~1.02, not 0.41, on a
   mechanism the feasibility screen itself describes as *"compressed post-2022, with 12-month
   rolling VRP episodes turning negative in 2023-24."*
2. **Kill switch 1 is now much more likely to fire.** It is already registered: *"if OPT-015 and
   OPT-019 jointly establish that index VRP is dead net of costs in the current regime, cancel
   Wave 2's short-vol block."*
3. **Zero survivors is a materially more probable outcome, and remains pre-registered as
   legitimate.** It is not a reason to revisit this amendment.
4. **The prohibited-responses list is reaffirmed** and is the operative risk here: do not re-run a
   candidate with adjusted parameters; do not loosen a criterion after seeing a number; do not
   reclassify a tested candidate as a diagnostic to dodge the trial count; **do not revise N or the
   window downward to lower this bar.**

---

## 7. Honesty block

Nothing in this amendment observed a strategy return. The correction runs in the **unfavourable**
direction -- the bar went up and the window shrank -- which is the opposite of a result-motivated
change, but that does not make it self-validating: it is registered because it is **measured**, and
it would stand equally if it had run the other way.

**Known-uncertain, stated plainly:**

- **N is uncertain across 179-621.** The dominant driver is the decision to count parameter sweeps
  at **full cardinality** (~195 trials), which overrides a prior ledger annotation
  (`trial_count_treatment=NOT_PROMOTED_no_selection_trial_added`) on 78 RAMP robustness rows. The
  reasoning: all 78 Sharpes were observed, and DSR counts **draws**, not promotions. That is a
  genuine methodology choice previously decided the other way, and it is recorded as overridden.
  191 identical-spec reruns are counted as **0** (apparatus correction, not search); counting them
  gives N = 621.
- **The only identified over-count** is ~2-6 possible duplicates between the off-ledger 36-chain
  and registry `RAMP-V11` / `V02+V05` rows. Every other identified uncertainty **under**-counts.
- **The window figure is a working assumption in one respect:** it takes the ThetaData coverage
  boundary as the explanation for the pre-2017 gap. That explanation is documented by the vendor
  and matches our data exactly, but ThetaData support has **not** been asked to confirm it. Doing so
  is free and would settle it.

*End of Amendment A2. Ledger action: append A2 rows to the DSR/window entries of slate v1.1,
feasibility screen v2, handoff spec v2 and amendment A1; retire the superseded null-best-of-N
tables in all four. Candidate definitions untouched.*
