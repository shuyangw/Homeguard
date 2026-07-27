# PRE-REGISTRATION: Equal-Weight Combination Spec (COMBO-1)

**Locked:** 2026-07-26, BEFORE any component of the coming wave exists.
**Status:** LOCKED. Amending any section below invalidates this pre-registration.

## Why this is declared now and not later

A combination is the only honest route past a bar of ~1.18 using components
whose individual if-true Sharpes sit around 0.5-0.8. If several near-uncorrelated
components each carry a true Sharpe well under the bar, an equal-weight book of
them can clear it while every component fails alone.

That argument is also the easiest thing in quantitative research to abuse. Wait
until component results are in, pick the ones that worked, and "the combination"
becomes cherry-picking with extra steps. **The declaration order is the entire
control.** This document exists before a single component has been written, let
alone run, which is why it specifies a RULE rather than a list of components.

## The rule (fixed, no discretion)

**Membership.** Every spec in the coming wave that satisfies BOTH:
1. it cleared the statistical-viability screen at proposal time
   (`screen_spec(...).viable is True`), AND
2. it was actually run to a walk-forward OOS return series.

No other inclusion or exclusion is permitted. In particular: a component is NOT
dropped for having performed badly. A component that fails its own gate stays in
the book. This is the clause that makes the spec falsifiable rather than
decorative.

**Weights.** Equal weight, 1/K over the K members. Not risk-parity, not
inverse-vol, not optimized, not rebalanced toward winners. Equal.

**Combination method.** Sum the member OOS daily return series over the union of
their dates, treating a date where a member has no position as that member
contributing 0.0 for that date, then divide by K. The result is one daily return
series, gated exactly like any other spec.

**Rebalance.** None. The 1/K weights are static for the whole OOS period.

**Trial count.** ONE trial, counted in the wave's total.

**Minimum K.** If K < 3 the combination spec is VOID and is not run, because a
one or two component "book" tests diversification not at all. A void outcome is
reported as void; it does not license a substitute rule.

## Gate

The standard combined statistical gate, identical to every other spec: positive
deflated Sharpe clearing PSR / DSR / PBO at the N prevailing when it runs, plus
survival of the mandatory 1.5x cost-stress leg. The S&P leg is book-level
context and is not gating.

No special allowance is made for this being a portfolio. It faces the same bar.

## Registered prediction

Recorded in advance so the result cannot be retrofitted.

**Predicted: FAIL.** The diversification argument is real but requires the
components to carry genuine positive expectancy AND low mutual correlation. The
campaign to date has produced no evidence of positive expectancy in any FX
component at any frequency tested. Averaging several zero-edge series produces a
zero-edge series with lower variance, which raises nothing: Sharpe is
scale-invariant, so shrinking the noise does not manufacture a signal.

The combination therefore only helps if at least some components have real,
positive, near-uncorrelated edge. If none do, this spec fails and its failure
is informative: it closes the "but a portfolio of them would have worked"
objection that would otherwise survive every individual negative.

**What would make me wrong:** two or more components with genuinely positive OOS
expectancy and pairwise correlation below roughly 0.3. That is a real
possibility for event-time components, which trade at different instants than
everything else in the book, and it is the reason this spec is worth one trial.

## What this spec cannot do

It cannot rescue a wave of negatives. If every component has zero or negative
expectancy, no weighting of them clears the bar, and no result from this spec
should be read as evidence that the components were individually better than
they measured.

## Falsifier

If the combination's OOS Sharpe is not materially above the mean of its
components' OOS Sharpes, the diversification premise is absent in this data and
the combination approach is dead for FX, not merely this instance of it.
