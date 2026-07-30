# Options Doc-Chain Amendment A3 — Mark Source for OPT-047 / OPT-030 (2026-07-30)

**Date:** 2026-07-30
**Trigger:** the **D-047/030 integrity gate FAILED** on measurement (M7 surface validation,
`20260730_m7_surface_validation.md`). The gate's own registered consequence requires a different
mark source before either candidate runs. This amendment supplies it.
**Amends:** `2026-07-25_options_slate_cc_handoff_spec_v2.md` (§3 P1, §4 M2/M7, §6.1 OPT-047,
§6.3 OPT-030) · `2026-07-25_options_slate_feasibility_screen_v2.md` (OPT-030, OPT-047 entries)
**Instrument:** logged amendment per the chain's registered rule — deviations require a logged
amendment **before the affected test runs**; superseded text is retired, never overwritten in place.
**Status:** still BLIND. **No strategy backtest has run. No P&L has been observed.** This change is
driven by a pre-registered integrity gate failing on a threshold that was **committed to git before
the first fit ran** (`6271424`, docs-only, 132 lines, 2026-07-30 08:18 ET), verified independently.

---

## 1. This is the pre-registration executing, not a workaround

The registered text of D-047/030 (handoff spec v2 §5, Group B):

> **Integrity gate — if the surface diverges materially at 0.05Δ, both candidates need a
> different mark source before running.**

The gate failed. Supplying a different mark source is the **registered consequence**, specified in
advance of the measurement. It is not engineering around a bad result.

What would be a workaround, and is explicitly **not** being done: relaxing the gate threshold,
re-fitting the surface until it passes, or quietly marking off `iv_smooth` anyway.

---

## 2. The measurement

`iv_smooth` vs owned raw quotes at 0.05 delta, thresholds registered before fitting:

| | inside quoted bid-ask IV band (need **≥90%**) | median divergence (need **≤1.5** vol pts) | verdict |
|---|---|---|---|
| SPY | **27.5%** | 0.154 | **FAIL** |
| QQQ | **55.1%** | 0.122 | **FAIL** |

**The two columns reconcile, and the reason matters.** Median divergence *passes* comfortably —
the surface sits ~0.12–0.15 vol points from the market. It fails the band test because the median
quoted IV band at 0.05 delta is only **0.09–0.35 vol points**: tighter than a 5-parameter form can
thread. **The surface is close to the market; the market is quoted more finely than the fit can
follow.**

Corroborating, from the same battery:
- Far-wing fit error is heavy-tailed: **p99 = 17.1 (SPY) / 23.7 (QQQ) vol points** in the
  `|Δ| < 0.05` bucket, with **~31% of contracts extrapolated** beyond the fitted range. This is
  the region OPT-047 reads and it is the least trustworthy region of the surface.
- Raw two-sided quotes exist at 0.05 delta in **100.0% of sessions** (median `spread_rel` 0.0206).

---

## 3. What changes

### 3.1 Marks — OPT-047 and OPT-030 (binding)

| Superseded | Replacement |
|---|---|
| P1 low-delta rule as applied to marks: *"for `|target_delta| < 0.10`, read delta and IV from the smoothed surface (`iv_smooth`)"* | **OPT-047 and OPT-030 mark off raw quote mid** — `options_chain_eod.mid`, valid rows only (`quote_valid`), crossed/zero-bid excluded per V3. This is **M2's general rule**, which these two candidates are hereby returned to. Trade prints (`close`/`vwap`) remain prohibited as marks, unchanged |

`iv_smooth` **may not** be used as a mark for these candidates. Where a valid quote is absent, the
position is not marked and the session is **reported as unmarked** — never filled from the surface.

### 3.2 Strike selection — OPT-047 and OPT-030 (binding)

Selection is a **different measurement** from marking and was validated separately. At OPT-047's
actual operating point (3-month, 0.05 delta) surface **extrapolation was 0.0%**, and strike-choice
agreement with shipped greeks is 74.3% (SPY) / 87.2% (QQQ).

- Selection uses **`delta_smooth`** where the surface is **non-extrapolated** at the selection point.
- Where it **is** extrapolated, selection falls back to `delta_shipped`, and the row is
  **flagged**.
- **Every trade logs which source selected its strike**, and results report the split. If the
  fallback share is material, that is a reportable limitation, not a footnote.

**Rationale for treating selection and marking differently:** the gate that failed measured
*marks* (IV against the quoted band). Selection was measured separately and passed at the
operating point. Splitting them is evidence-based, not convenient — but the 74.3% SPY agreement is
**not overwhelming**, and where the two sources disagree there is no ground truth. Recorded as a
known limitation.

### 3.3 OPT-027 — clarification, no change

OPT-027's long leg is at **0.10 delta exactly**. P1's rule binds at `|target_delta| < 0.10`, so
0.10 is **not** below the threshold and the rule **does not bind**. OPT-027 uses the standard M2
mark and is **unaffected by this amendment**. Recorded to remove ambiguity, not to move the candidate.

### 3.4 OPT-006 — remains BLOCKED, deliberately unresolved

OPT-006's 12–18 month long leg **exceeds M7's registered 400-day cap**. Three paths exist (widen
the cap and re-validate at 365–550 DTE; mark the LEAPS leg off raw quotes; defer). **None is chosen
here.**

OPT-006 is a **Wave 2** candidate, so this does not gate Wave 1, and the choice is not obvious:
LEAPS quotes are the widest on the slate ($0.30–1.00), which is precisely the case where a smoothed
mark is *most* wanted and a raw-quote mark *least* reliable. **The cap may not be widened without
re-running M7's full validation battery at 365–550 DTE.** Deciding under Wave-1 time pressure would
be the wrong call; it is deferred to the Wave-2 gate.

---

## 4. A registered finding that this amendment does NOT act on

P1's low-delta rule was justified by *"deep-OTM IV error"* in the vendor's dividends-off
Black-Scholes greeks. **Measurement contradicts that rationale.** Median
`|delta_smooth − delta_shipped|` on SPY 2024-01:

| `|delta|` bucket | median discrepancy |
|---|---|
| < 0.05 | **0.0008** |
| > 0.65 | **0.0187** |

Rising monotonically — the shipped-greek discrepancy is **smallest at low delta and largest deep
ITM**, the opposite end from the one the rule targets.

**No slate-wide relaxation of P1 is made here.** One root, one month, one comparison against a
surface that itself failed its band test is not sufficient evidence to unwind a rule that governs
every candidate. The finding is **registered** so that it is not rediscovered, and so that any
future amendment to P1 starts from it rather than from the original rationale.

---

## 5. Consequences

- **Wave 1 is unblocked at 9 candidates.** OPT-047 has a valid mark source; OPT-027 never needed
  one; the remaining seven were never affected.
- **OPT-030 (Wave 3) is unblocked** on the same basis.
- **OPT-006 (Wave 2) stays blocked** pending §3.4.
- M7 is **retained and remains useful** — for strike selection at non-extrapolated points, for
  curvature, and prospectively for OPT-006's LEAPS leg if the cap question is resolved. It is
  **not** retired by this amendment; its validated scope is narrowed to what it measurably supports.
- The hurdle registered in **A2 (1.02)** is unchanged. Nothing here alters N or the window.

## 6. Honesty block

This amendment was written **after** a gate failed and **because** it failed — which is exactly the
circumstance in which specification changes are most suspect. Three guards against that reading,
stated so they can be checked:

1. The consequence applied is the one the gate itself registered in advance, **verbatim**.
2. The failed threshold was **committed to git before the first fit ran** and was not retuned after
   the result was seen — independently verified.
3. The replacement mark source is **strictly more conservative**: raw two-sided quotes are the
   market's own prices, not a model's output. This amendment reduces model dependence rather than
   adding a degree of freedom.

**What remains genuinely uncertain:** OPT-047 marks off quotes in the region where quotes are
widest in relative terms, so its cost sensitivity is the real test and the ±50% band must be read
as load-bearing, not decorative. Strike-selection disagreement between sources (~26% on SPY) has no
ground truth. And M7's far wing is heavy-tailed exactly where this candidate operates — which is
*why* marks come from quotes, but it also means the surface cannot serve as an independent check on
those marks.

*End of Amendment A3. Ledger action: append A3 rows to OPT-047, OPT-030, OPT-027 (clarification)
and OPT-006 (blocked); retire the P1-as-applied-to-marks text for 047/030 in handoff spec v2.
Candidate definitions, parameters, priors, falsifiers and wave assignments are otherwise untouched.*
