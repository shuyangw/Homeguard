# Forward-Paper Evaluation Protocol (LOCKED)

**Locked:** 2026-07-26, before the first observation on 2026-07-31.
**Applies to:** `fx-month-end-fix`, `fx-quarter-end-fix`
(`config/forward_paper/specs.yaml`).
**Status:** LOCKED. Amending any rule below invalidates observations gathered
under it; an amendment starts a new log, it does not revise this one.

## Why this is locked before observation 1

A forward log whose evaluation rule is written after the observations start is a
garden of forking paths with a multi-year fuse. With 12 and 4 events a year, an
undeclared rule would allow years of ambiguity to be resolved in whichever
direction the accumulated data happened to point. Every threshold below is fixed
now, when nothing is known.

---

## 1. Executability finding (determined before locking)

**The registered construction is NOT executable in this account.**

The specs enter "in the direction implied by hedge rebalancing", which is
directional and takes both signs. A cash account cannot hold a negative foreign
currency balance, because that is precisely a leveraged spot position and the
account has no ECP status. So for each pair only ONE signal direction is
executable: the one ending long the foreign currency, funded from USD cash.

Roughly half of all events are therefore unexecutable, which halves the trade
count and raises the capital each spec would need:

| spec | as registered | cash-executable subset |
|---|---:|---:|
| fx-month-end-fix | $163,855 | **$344,718** |
| fx-quarter-end-fix | $70,932 | **$136,911** |

This makes the FX closure *more* robust, not less: the survivors are further out
of reach than the resolution stated.

Consequence for this protocol: **both bases are marked on every event.** The
construction is not quietly narrowed to whatever the account could have done,
which would silently substitute a different strategy for the registered one.

---

## 2. Marking procedure

**Data source.** Dukascopy bid/ask, retrieved post-hoc. No live capture is
required; these are paper marks, and post-hoc retrieval of a timestamped quote
is not a lookahead because the entry and exit timestamps are fixed by the spec
and independent of the outcome.

**Prices are BID/ASK-CROSSED, never mid.** A long entry pays the ask and exits
at the bid; a short does the reverse. Marking at mid would flatter every
observation by exactly the quantity the whole campaign found decisive.

**Timestamps.** Entry 14:00 UTC, exit 16:05 UTC, as locked in `specs.yaml`. If a
quote is missing at the exact minute, use the last quote at or before it and
record `notes="stale_quote:<seconds>"`. If no quote exists within 15 minutes,
the event is recorded with `signal=0, return_bps=0.0,
notes="no_market"` -- it is NOT dropped, because dropping events is how a log
quietly conditions on outcome.

**Signal.** Sign of the month's (or quarter's) relative equity return per the
spec, computed from data available at 14:00 UTC on the event date.

**Who writes it.** `src.backtesting.validation.forward_paper.record_observation`,
which refuses any date at or before the spec's lock and refuses to re-record an
event.

---

## 3. Cost bases: both marked, every event

Every event is recorded twice, under two explicitly different cost assumptions,
so the log answers two different questions without either contaminating the
other.

| basis | cost assumption | question it answers |
|---|---|---|
| **mechanism** | threshold notional ($100k+/order, so the $2 minimum does not bind) plus measured hour-of-week spread | Is the effect real? |
| **deployable** | actual current capital, 6 concurrent orders, commission floor applied, long-foreign-only subset | Could this account have traded it? |

Recorded as two rows with `spec` suffixed `:mechanism` and `:deployable`.

The mechanism basis is explicitly **not** a claim about tradability. It exists
because "the effect is real but we cannot reach it" and "the effect is not
there" are different findings with different consequences, and a single-basis log
cannot distinguish them.

---

## 4. Promotion and kill rule (pre-registered)

A fixed-n frequentist test would need years before it says anything. This uses a
Bayesian sequential rule with the boundaries fixed now.

**Model.** Per-event net return in bps, mechanism basis, treated as i.i.d.
Normal with unknown mean `mu` and unknown variance. Prior on `mu`: Normal
centred at **0** with standard deviation **6.0 bps**, the pre-registered if-true
edge for `fx-month-end-fix`. Centring at zero rather than at the if-true estimate
is deliberate: the prior must not encode the conclusion.

**Looks.** After events 8, 16, 24, 32, and every 8 thereafter. No unscheduled
looks.

**Boundaries, both bases required for promotion:**

| decision | condition |
|---|---|
| **PROMOTE** | `P(mu > 0) >= 0.95` on the mechanism basis **AND** `P(mu > 0) >= 0.90` on the deployable basis |
| **KILL** | `P(mu > 0) <= 0.25` on the mechanism basis |
| **CONTINUE** | otherwise |

Promotion requires the deployable basis too, so a mechanism that is real but
untradeable at this account size is recorded as exactly that: confirmed and
shelved, not promoted.

**Calendar kill.** If no PROMOTE or KILL boundary is reached by **2028-12-31**,
the spec is retired as inconclusive. A log that can run forever on ambiguity is
not evidence, it is a habit.

**Trial accounting.** Forward observations consume no trials, at any point,
under any outcome. They contact no historical sample and exert no selection
pressure on it. A PROMOTE would trigger a fresh pre-registration for live
deployment, which is where trials would be counted.

---

## 5. Correlation logging

Quarter-ends **are** month-ends: all 4 quarter-end events each year are also
month-end events. The two specs therefore share a quarter of the month-end
sample, and treating their observations as independent would overstate the
evidence by double-counting.

Both specs' per-event returns are logged jointly on shared dates. Any decision
that considers both specs together must use the union of distinct event dates,
never the sum of the two counts.

---

## 6. What could invalidate this protocol

Stated so the protocol is falsifiable rather than merely followed:

- The executability finding in Section 1 rests on the claim that a cash account
  cannot hold a negative foreign balance. If that is wrong for this account, the
  deployable basis is too strict and must be re-derived, and observations already
  marked under it are re-marked (the mechanism basis is unaffected).
- If the equity-return signal proxy proves unavailable at 14:00 UTC in practice,
  the spec has a data defect and the log stops rather than substituting a
  different proxy.
- If Dukascopy coverage is absent for an event, Section 2's `no_market` rule
  applies. Systematic absence (more than 3 events) is a data defect, not a
  result.
