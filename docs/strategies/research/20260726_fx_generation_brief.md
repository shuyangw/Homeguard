# Generation Brief: FX Wave 3 Slate (for a FRESH context)

**Purpose:** produce a ~50-spec candidate slate for the FX intraday / event-time
wave, generated WITHOUT sight of any completed run's results.

## How to run this

Start a **new session**. Do not run this in a context that has read campaign
results, report files, session logs, or the catalog tracker. The generator's
entire permitted view of the campaign is one file:

    docs/strategies/research/20260726_fx_generation_ledger.md

Rebuild it first if stale: `PYTHONPATH=$(pwd) python scripts/strategy/build_generation_ledger.py`

### Files the generator MUST NOT read

- `docs/strategies/FX_60_CATALOG_TRACKER.md` (Notes column is full of OOS scores)
- anything under `docs/reports/`
- `docs/strategies/research/*_results.md`, `*_resolution.md`, `*_regate.md`,
  `*_synthesis.md`, `*_cost_sensitivity_regate.md`
- `docs/progress/*` session logs
- `output/experiments.duckdb`
- `~/Downloads/20260726_fx_campaign_next_steps.md` and any prior slate proposal

This list is the point of the exercise. The previous slate proposal was written
in a context that had just read every result, and its author said so. Conditioning
candidate choice on results turns a pre-registration into a rationalisation.

## What to produce

Roughly 50 specs. For each:

1. **Mechanism first.** Why does this edge exist, who is on the other side, and
   why does it survive arbitrage? A rule without a mechanism is a data-mining
   result waiting to happen. State the mechanism BEFORE the rule.
2. **The rule**, fully specified: universe, entry, exit, holding period,
   rebalance, and every parameter at a fixed value. No ranges, no sweeps.
3. **Viability screen result.** Required, not optional:
   ```
   from src.backtesting.validation.viability import screen_spec
   screen_spec(name=..., trades_per_year=..., gross_edge_bps=...,
               per_trade_vol_bps=..., pairs=[...], hours_of_week=[...],
               sr_zero=<from the ledger>)
   ```
   State `gross_edge_bps` and `per_trade_vol_bps` honestly, with their source
   (literature, first principles, or a measurement). Do NOT state a cost: the
   screen computes it from the measured spread surface for your pairs and hours.
   A spec that does not clear the bar is not thereby dead, but it must be routed
   to the forward-paper queue or the combination spec rather than to a trial.
4. **Falsifier.** The observation that would kill the mechanism.
5. **Most likely spurious reason.** How this could look good and be wrong.
6. **Kill conditions.** What ends it during testing.

## Hard constraints

- **Frequencies**: 1m and any aggregation of it, plus daily. Nothing sub-minute.
- **Execution**: spread-TAKER only. Liquidity provision is not backtestable on
  the data held and belongs to a forward track, never a historical trial.
- **Events**: US CPI / NFP / FOMC have authoritative, DST-correct, validated
  timestamps (FOMC from 2013). Non-US central banks do NOT. A spec depending on
  ECB / BoE / BoJ / BoC / SNB / RBA / RBNZ event times cannot be run and must
  either be dropped or scoped to the US calendar.
- **Cost caveats**: EURGBP and GBPJPY are unmeasured and take a conservative
  fallback; any pair outside the measured 25 gets a flat hourly shape. A spec
  leaning on those pairs inherits real cost uncertainty and should say so.
- **Data**: no options-implied, no order-book, no order-flow, no consensus
  economic forecasts. None of it is held.
- **ML slots are NOT runnable**: the meta-label harness (triple-barrier +
  feature pipeline) does not exist. Do not propose specs that need it
  unless you are explicitly proposing to build it first.

## Slot arithmetic, so the slate size is a deliberate choice

The ledger shows 43 OPEN slots plus 4 TESTED-NAIVE-ONLY. Not all are runnable:

| capability | open slots | runnable now? |
|---|---:|---|
| INTRADAY | 21 | yes, engine exists |
| READY | 7 | yes |
| OHLC | 4 | yes |
| BRACKET | 3 | yes |
| SPREAD | 1 | yes |
| ML | 6 | **NO** -- the meta-label harness is not built |
| DATA | 1 | **NO** -- the data is not held |

So **40 runnable catalog slots** (36 open + 4 naive-only re-forms). A ~50 spec
slate therefore needs roughly **10 novel specs** outside the 60-catalog. That is
a legitimate and expected part of the exercise, not a shortfall: the catalog is
a starting inventory, not a ceiling. But do not pad toward 50 with weak ideas.
Fewer, better-reasoned candidates beat a full slate of thin ones, because every
spec raises the bar for every other spec in the wave.

If, after honest generation, you have materially fewer than 50 specs that clear
the viability screen and carry a real mechanism, SAY SO and submit the smaller
slate. Reporting a short slate is a correct outcome; inventing filler is not.

## Diversity requirement

The tested-and-failed families in the ledger are broad. Do not fill 50 slots
with variations on one idea. Budget the slate across distinct mechanism
families, and say what the budget is. Repetition inflates N while adding little
independent information, which is the worst possible trade against a bar that
rises with N.

## Already locked, do not re-derive

The equal-weight combination spec is ALREADY pre-registered at
`docs/strategies/research/20260726_fx_combination_spec_prereg.md`, before any
component of this wave exists. Its membership rule is every spec in the wave
that cleared the viability screen and was run, equal weighted. You do not choose
its members, and you must not propose a competing combination rule.

## Output

A single pre-registration document, committed BEFORE anything is run, listing
every spec with all six fields above plus the slate's mechanism-family budget
and the ledger's SR_zero as the bar each will face.

## Reminder on what success looks like

Zero survivors is a valid and complete outcome. The objective is a set of
honestly-specified, mechanism-grounded hypotheses that CAN clear the bar if
true, not a set that will pass. Do not tune toward passing; there is nothing
here to tune against, by design.
