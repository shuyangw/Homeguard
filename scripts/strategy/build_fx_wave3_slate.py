"""Screen the Wave 3 slate and emit the pre-registration document.

Usage: PYTHONPATH=$(pwd) python scripts/strategy/build_fx_wave3_slate.py
"""
from pathlib import Path

from src.backtesting.validation.viability import (screen_spec, expected_cost_bps,
                                                  if_true_sharpe)
from scripts.strategy.fx_wave3_slate_defs import SPECS

ROOT = Path(__file__).resolve().parents[2]

SR_ZERO = 1.1807
N_PRIOR = 141

FAM_NAMES = {
    "F1": "US scheduled-event time (CPI / NFP / FOMC)",
    "F2": "Session and time-of-day segmentation",
    "F3": "Benchmark fixing and rebalancing flow",
    "F4": "Intraday breakout and volatility expansion",
    "F5": "Intraday mean reversion (taker-side inventory residual)",
    "F6": "Cross-sectional and dollar-factor structure",
    "F7": "Lead-lag and cross-market propagation",
    "F8": "Cross-asset metals-FX linkage",
    "F9": "Carry and swap-aware forms",
    "F10": "Calendar and liquidity-regime effects",
}

results = []
for s in SPECS:
    r = screen_spec(name=s["name"], trades_per_year=s["T"],
                    gross_edge_bps=s["edge"], per_trade_vol_bps=s["vol"],
                    pairs=s["pairs"], hours_of_week=s["hours"], sr_zero=SR_ZERO)
    legs = s["legs"]
    cost1 = expected_cost_bps(s["pairs"], s["hours"])
    cost_n = cost1 * legs
    sharpe_n = if_true_sharpe(s["T"], s["edge"], cost_n, s["vol"])
    sharpe_15 = if_true_sharpe(s["T"], s["edge"], 1.5 * cost_n, s["vol"])
    results.append({**s, "screen": r, "cost1": cost1, "cost_n": cost_n,
                    "sharpe_n": sharpe_n, "sharpe_15": sharpe_15,
                    "viable": sharpe_n > SR_ZERO and sharpe_15 > SR_ZERO})

viable = [r for r in results if r["viable"]]
routed = [r for r in results if not r["viable"]]

# --- integrity checks -------------------------------------------------------
slots = [r["slot"] for r in results if r["slot"] != "NOVEL"]
assert len(slots) == len(set(slots)), f"duplicate catalog slot: {slots}"
RUNNABLE = {"1","2","3","4","5","7","8","9","10","11","13","14","15","17","21","22",
            "23","24","25","26","28","31","32","34","36","38","40","41","43","44",
            "45","46","47","54","56","57","58","59","60"}
missing = RUNNABLE - set(slots)
extra = set(slots) - RUNNABLE
print(f"specs: {len(results)}  catalog: {len(slots)}  novel: {len(results)-len(slots)}")
print(f"viable: {len(viable)}  routed: {len(routed)}")
print(f"uncovered runnable slots: {sorted(missing) or 'none'}")
print(f"slots proposed outside runnable set: {sorted(extra) or 'none'}")

fams = {}
for r in results:
    fams.setdefault(r["fam"], []).append(r)

L = []
A = L.append
A("# FX Wave 3 Slate -- Pre-Registration (intraday / event-time axis)")
A("")
A("**Date:** 2026-07-26  ")
A(f"**Bar every spec faces:** SR_zero = **{SR_ZERO:.4f}** annualized, from the "
  f"generation ledger at N = {N_PRIOR} prior trials plus a {len(results)}-spec slate.  ")
A("**Status:** pre-registered. Committed BEFORE any spec in this slate is run.")
A("")
A("## Provenance and blindness")
A("")
A("Generated in a fresh context whose only permitted campaign input was")
A("`docs/strategies/research/20260726_fx_generation_ledger.md`. No results file,")
A("report, tracker, session log or experiment registry was read. Two disclosures:")
A("")
A("1. **A result leaked through the environment.** The session-start git log in the")
A("   system prompt contained the commit subject `fb169df test(fx): #20 London")
A("   Breakout re-gate on corrected apparatus -- FAIL, cost-robust`. That is a")
A("   verdict for catalog slot #20. It was not sought and arrived before the brief")
A("   was read. Slot #20 is excluded from this slate.")
A("2. **The leak exposed a real defect, now fixed.** The ledger listed #20 as OPEN.")
A("   `build_generation_ledger.py` mapped gate grades with")
A("   `_GRADE.get(cells[6], \"OPEN\")`, so any unrecognized grade string silently")
A("   became OPEN. The tracker holds `'FAIL (cost-robust)'` (with a space) while the")
A("   map had `'FAIL(cost-robust)'` (without), so a tested-and-failed slot was")
A("   presented to the generator as an open one. The parser now raises on unknown")
A("   grades and the token was added. Rebuilt counts: OPEN 43 -> 42, TESTED 13 -> 14,")
A("   READY-open 7 -> 6. The corrected ledger marks #20 TESTED-FAIL independently of")
A("   the leak, so the exclusion does not rest on leaked information.")
A("")
A("Measurement discipline: `per_trade_vol_bps` is MEASURED from the held 1m data")
A("(2011-2026, unsigned dispersion only). `gross_edge_bps` is a literature or")
A("first-principles estimate stated at proposal time. **No signed effect,")
A("autocorrelation or continuation was measured before proposing** -- doing so would")
A("condition the pre-registration on the answer. Costs are computed by the screen")
A("from the measured hour-of-week spread surface; none is asserted by hand.")
A("")
A("## A correction to the screen, applied to every multi-leg spec")
A("")
A("`screen_spec` has no concept of legs: it averages the round-trip cost over the")
A("pairs named and charges it ONCE. A 2-leg spread pays two round trips and a 6-leg")
A("basket pays six. Left uncorrected the screen would flatter every relative-value")
A("spec in this slate. Each entry below therefore reports the raw screen output and,")
A("where legs > 1, a **leg-adjusted if-true Sharpe** computed with")
A("`if_true_sharpe(T, edge, legs * cost, vol)`. **The leg-adjusted figure is the")
A("authoritative one** and decides routing. This is a limitation of the screen worth")
A("fixing in `viability.py`, not a property of these specs.")
A("")
A("## Two gates, not one")
A("")
A("A spec earns a trial only if it clears the bar **and still clears it with costs")
A("at 1.5x the measured surface** (methodology Section 4). The second gate is not")
A("decoration: the high-trade-count intraday specs in this slate have net edges of a")
A("few tenths of a bp against ~0.6-1.0bps of cost, so a modest cost misestimate")
A("flips their sign. Specs that pass only at 1.0x cost are reported as routed, with")
A("the binding constraint named.")
A("")
A("## Self-audit of this document")
A("")
A("A first draft of this slate had 19 specs clearing the bar. Reviewing my own")
A("inputs before publishing, three faults were found and corrected DOWNWARD:")
A("")
A("- **A trigger threshold used as an expected edge.** Spec 35 (synthetic cross)")
A("  was given `gross_edge_bps = 3.0`, which is the entry threshold -- assuming full")
A("  capture of the divergence on every trade -- while its own spurious-reason field")
A("  said the true edge is approximately zero. Corrected to 0.5.")
A("- **Numbers contradicting their own prose.** Spec 14 (NOKSEK) was given an edge")
A("  of 30% of per-trade volatility while its text stated the two-leg cost would")
A("  erase it. Corrected from 9.0 to 5.0bps.")
A("- **Trade counts set to the maximum possible rather than the expected trigger")
A("  rate.** Twelve thresholded specs had `trades_per_year` set as if every pair")
A("  traded every day, inflating sqrt(T). Corrected to expected trigger rates.")
A("")
A("These corrections moved specs from viable to routed. That direction is the point:")
A("the failure mode this campaign is guarding against is an author tuning inputs")
A("until specs pass, and the only defence is to audit one's own numbers against")
A("one's own stated reasoning before the run, not after.")
A("")
A("## Slate summary")
A("")
A(f"- Specs pre-registered: **{len(results)}** ({len(slots)} catalog slots + "
  f"{len(results)-len(slots)} novel)")
A(f"- Clear the bar on their own if-true arithmetic: **{len(viable)}**")
A(f"- Cannot clear the bar even if entirely correct: **{len(routed)}** -> routed to "
  "the forward-paper queue / combination spec, NOT to standalone trials")
A("")
A("The runnable catalog inventory is 39 slots (35 open + 4 naive-only re-forms)")
A("after the #20 correction; all 39 are covered here, plus 10 novel specs.")
A("")
A("### Mechanism-family budget")
A("")
A("| family | mechanism | specs | clear bar |")
A("|---|---|---:|---:|")
for f in sorted(fams, key=lambda k: int(k[1:])):
    rs = fams[f]
    A(f"| {f} | {FAM_NAMES[f]} | {len(rs)} | {sum(1 for r in rs if r['viable'])} |")
A("")
A("No family exceeds 11 of 49 specs. The budget is deliberately weighted toward")
A("event-time and session structure, where the apparatus has authoritative")
A("timestamps and a measured hour-of-week cost surface, and away from daily factor")
A("families, which the arithmetic below shows cannot reach this bar at all.")
A("")
A("The families are not equally independent, and saying so matters more than the")
A("count: F4 (breakout) and F6 (cross-sectional momentum) both rest on slow")
A("information diffusion, and F5 and F2's reversion specs both rest on inventory")
A("absorption. Specs 23 and 27 differ mainly in their exit rule and are flagged in")
A("their own kill conditions as at risk of being one idea counted twice. Treating")
A("all 49 as independent evidence would overstate the slate's information content.")
A("")
A("### What the arithmetic says before anything is run")
A("")
A("Three structural results fall out of the screen, and they shape the slate more")
A("than any individual idea:")
A("")
A("1. **Daily G10 factor specs are arithmetically incapable of passing.** Published")
A("   net Sharpe for FX trend, carry and cross-sectional momentum is 0.3-0.6. Since")
A("   the if-true Sharpe is derived FROM that literature figure, it is by")
A("   construction below 1.18. Every such spec here is routed, not tested. This is")
A("   not pessimism, it is arithmetic: at ~40 trades/year no honest per-trade edge")
A("   reaches the bar.")
A("2. **Event-time drift specs fail on trade count, not on mechanism.** With only")
A("   ~30 US releases a year and measured post-event dispersion of 19-32bps, even a")
A("   generous drift estimate gives sqrt(180) * 0.4/22 ~ 0.26. The event calendar is")
A("   our best data asset and it still cannot support a standalone daily-frequency")
A("   verdict. Event specs survive only where the edge/vol ratio is structurally")
A("   large, not where the mechanism is merely real.")
A("3. **High-trade-count intraday specs clear the bar at measured cost and collapse")
A("   at 1.5x cost.** Specs 12, 15, 19, 24, 28 and 49 have if-true Sharpes of")
A("   1.24-2.36 at the measured surface and 0.31-1.03 at 1.5x. Their net edges are")
A("   a few tenths of a bp against ~0.6-1.0bps of cost, so they are tests of the")
A("   cost model as much as of the signal. Running them would spend six trials on")
A("   results that a plausible cost misestimate could reverse. They are routed.")
A("4. **What survives both gates is concentrated, price-insensitive, calendar-known")
A("   flow.** Only the month-end and quarter-end fix specs clear at 1.5x cost, and")
A("   they do so with headroom rather than marginally: spec 18 needs a 4.40bps gross")
A("   edge to clear at 1.5x cost and is proposed at 6.0bps, against the 10-20bps")
A("   Melvin-Prins document. They are rare (72 and 24 trades/year) but their")
A("   edge-to-volatility ratio is structurally large, which is the only thing that")
A("   works at this bar.")
A("")
A("### Specs that clear the bar (ranked by leg-adjusted margin)")
A("")
A("| # | slot | name | if-true SR | at 1.5x cost | margin |")
A("|---|---|---|---:|---:|---:|")
for r in sorted(viable, key=lambda x: -(x["sharpe_n"] - SR_ZERO)):
    A(f"| {r['id']} | {r['slot']} | {r['name']} | {r['sharpe_n']:.2f} | "
      f"{r['sharpe_15']:.2f} | +{r['sharpe_n']-SR_ZERO:.2f} |")
A("")
A("### Specs routed to forward-paper / combination (cannot clear the bar if true)")
A("")
A("| # | slot | name | if-true SR | at 1.5x cost | binding constraint |")
A("|---|---|---|---:|---:|---|")
for r in sorted(routed, key=lambda x: -(x["sharpe_n"] - SR_ZERO)):
    why = ("fails 1.5x cost gate only" if r["sharpe_n"] > SR_ZERO
           else "below bar outright")
    A(f"| {r['id']} | {r['slot']} | {r['name']} | {r['sharpe_n']:.2f} | "
      f"{r['sharpe_15']:.2f} | {why} |")
A("")
A("Routing is not a verdict on the mechanism. It states that a standalone")
A("historical trial cannot produce a passing number even if the thesis is exactly")
A("right, so spending a trial on it would raise the bar for every other spec while")
A("being incapable of clearing it. Per the locked combination pre-registration")
A("(`20260726_fx_combination_spec_prereg.md`), membership is every spec in this wave")
A("that cleared the screen and was run, equal weighted. Routed specs are not")
A("members and this document does not propose an alternative combination rule.")
A("")
A("---")
A("")
A("## The specs")
A("")

for f in sorted(fams, key=lambda k: int(k[1:])):
    A(f"# {f}. {FAM_NAMES[f]}")
    A("")
    for r in fams[f]:
        slot = f"catalog #{r['slot']}" if r["slot"] != "NOVEL" else "NOVEL (outside the 60-catalog)"
        A(f"## Spec {r['id']}: {r['name']}")
        A("")
        A(f"*{slot}* | family {r['fam']} | "
          f"**{'CLEARS BAR' if r['viable'] else 'ROUTED -- cannot clear bar'}**")
        A("")
        A(f"**1. Mechanism.** {r['mech']}")
        A("")
        A(f"**2. Rule.** {r['rule']}")
        A("")
        A("**3. Viability screen.**")
        A("")
        A("```")
        A("screen_spec(")
        A(f"    name={r['name']!r},")
        A(f"    trades_per_year={r['T']}, gross_edge_bps={r['edge']}, "
          f"per_trade_vol_bps={r['vol']},")
        A(f"    pairs={list(r['pairs'])},")
        A(f"    hours_of_week=<{len(r['hours'])} weekday hours>, sr_zero={SR_ZERO})")
        A("")
        A(f"-> {r['screen'].summary()}")
        if r["legs"] > 1:
            A(f"-> legs={r['legs']}: cost {r['cost1']:.2f} x {r['legs']} = "
              f"{r['cost_n']:.2f} bps RT")
            A(f"-> LEG-ADJUSTED if-true Sharpe {r['sharpe_n']:.2f} vs bar "
              f"{SR_ZERO:.2f}")
        A(f"-> at 1.5x cost ({1.5*r['cost_n']:.2f} bps RT): if-true Sharpe "
          f"{r['sharpe_15']:.2f} -> "
          f"{'CLEARS' if r['sharpe_15'] > SR_ZERO else 'FAILS cost gate'}")
        A("```")
        A("")
        A(f"- `gross_edge_bps` = {r['edge']} -- {r['edge_src']}")
        A(f"- `per_trade_vol_bps` = {r['vol']} -- {r['vol_src']}")
        A("")
        A(f"**4. Falsifier.** {r['falsifier']}")
        A("")
        A(f"**5. Most likely spurious reason.** {r['spurious']}")
        A("")
        A(f"**6. Kill conditions.** {r['kill']}")
        A("")

A("---")
A("")
A("## Standing constraints honoured")
A("")
A("- Frequencies: 1m, aggregations of 1m, and daily only. Nothing sub-minute.")
A("- Execution: spread-TAKER throughout. No spec assumes liquidity provision.")
A("- Events: US CPI / NFP / FOMC only (FOMC 2013+). No spec depends on ECB, BoE,")
A("  BoJ, BoC, SNB, RBA or RBNZ event times.")
A("- No options-implied, order-book, order-flow or consensus-forecast data is used.")
A("- No ML slot is proposed: the triple-barrier meta-label harness does not exist,")
A("  so catalog slots 48-53 are deliberately left unfilled. Slot 55 (USDCNH PBOC")
A("  fix) is also unfilled: the fix data is not held.")
A("- Every parameter is fixed at a stated value. No ranges, no sweeps.")
A("- Cost caveats carried explicitly: spec 35 leans on EURGBP, which is unmeasured")
A("  and takes the 4.0bps conservative fallback (note the derived-cross table in")
A("  `costs/fx.py` is NOT consulted by `fx_round_trip_bps_at`, so the fallback is")
A("  what the screen actually charged). Specs on Nordic and EM pairs inherit the")
A("  measured-but-wide levels, and any pair outside the measured 25 gets a flat")
A("  hourly shape.")
A("")
A("## Trial accounting")
A("")
A(f"- Prior trials: {N_PRIOR}")
A(f"- Specs pre-registered here: {len(results)}")
A(f"- Specs that will consume a trial (cleared the screen): **{len(viable)}**")
A(f"- Specs routed without consuming a trial: {len(routed)}")
A("")
A("The bar quoted throughout is computed at N + 50 as the ledger specifies, which is")
A("conservative relative to the smaller number of trials actually to be consumed. If")
A("only the cleared specs are run, the bar should be RECOMPUTED at the true N before")
A("any verdict is issued -- and recomputing it downward after seeing which specs")
A("passed would be exactly the gate-tuning this campaign has ruled out. Fix N from")
A("the pre-registered intent, not from the outcome.")
A("")

out = ROOT / "docs/strategies/research/20260726_fx_wave3_slate_prereg.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(L)} lines)")
