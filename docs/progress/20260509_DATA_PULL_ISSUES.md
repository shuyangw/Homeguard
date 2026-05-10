# Data Pull Issues - 2026-05-09

Three failures during the Tasks 15-19 operational pull, with diagnosis and what should have prevented each.

## Issue 1: Trades_ES_MES rejected with `402 account_insufficient_funds`

### What happened
Submitted `trades` schema for `ES.v.0 + MES.v.0`, 5 years (2021-01-01 -> 2026-02-22). Databento returned `BentoClientError: 402 account_insufficient_funds`.

### Cost (queried via `metadata.get_cost` after the fact)
| Scope | Cost |
|---|---|
| Full 5y, ES + MES | **$1040.68** |
| ES only, full 5y | $601.52 |
| ES + MES, 1y only | $89.30 |

### Why it failed
The Databento subscription covers OHLCV (1m, 1d), `definitions`, `statistics`, `status`, and the **MBP-1 free sliver** (last 6 months of 4 symbols). It does **not** cover the `trades` schema at the volume we requested. `trades` is priced separately and the MBP-1 free tier does not have an analogous free window for trades.

### Root cause
Planning assumption. The original spec (`docs/superpowers/specs/2026-05-09-data-validation-framework-and-additional-pull-design.md`) and the plan (`docs/superpowers/plans/2026-05-09-data-validation-framework-and-additional-pull.md`) both assumed the trades pull was subscription-covered. No cost-check gate before `client.batch.submit_job()`.

### What would have caught this
A `metadata.get_cost()` precheck for each new section before submitting, with an explicit cost-acknowledgment gate:

```python
cost = client.metadata.get_cost(dataset, schema, symbols, stype_in, start, end)
if cost > BUDGET_THRESHOLD:
    confirm or skip
_submit(...)
```

### Status
Deferred. The validation framework, derivations, status events, and Eurodollar daily data all landed without this. Trades only enables a specific Adaptation A signal (order imbalance proxy) — not needed for the strategies currently in flight. Subscription expires 2026-06-01, so any decision needs to be made before then.

---

## Issue 2: B_ED_daily rejected with `422 symbology_invalid_request`

### What happened
Submitted `ohlcv-1d` for `ED.FUT` (parent symbology) for Eurodollar futures, 2010-06-06 -> 2023-12-31. Databento returned `422 symbology_invalid_request: Could not resolve smart symbols: ED.FUT`.

### Why it failed
CME phased out Eurodollar futures by end-of-trading 2023. Databento doesn't accept `ED.FUT` in their current symbology. The legacy CME root symbol is `GE` (not `ED`):
- `ED` was the colloquial market name (Eurodollar)
- `GE` was always the CME root ticker

Symbology resolution check confirmed:
| Symbol | stype_in | Result |
|---|---|---|
| `ED.FUT` | parent | `422 symbology_invalid_request` |
| `GE.FUT` | parent | 1238 contracts resolved |
| `EDU3` | raw_symbol | not found |
| `GEU3` | raw_symbol | resolves to 1 instrument |
| `ED.c.0` | continuous | not found |
| `GE.c.0` | continuous | 42 mappings |

### Root cause
Used the conversational name (`ED`) instead of the exchange root (`GE`). The plan inherited this naming from the original `01_ADDITIONAL_DATA_ACQUISITION.md` doc which referred to "Eurodollar (ED)" colloquially. Nobody (including me, when writing the plan) cross-referenced Databento's symbology table.

### Fix
Patched `scripts/data/databento_batch_submit.py` to use `GE.FUT`. Resubmission succeeded at `$0.00`. Conversion landed 1.2M rows under `H:/Stock_Data/futures_per_contract_daily/root=ED/`. Note: storage path retains the colloquial `root=ED` for human-readability, but the upstream Databento symbol is `GE`.

### What would have caught this
A symbology probe before submission, using `client.symbology.resolve()` to verify each parent symbol returns at least 1 instrument before submitting the batch job. Same pattern as the cost check above — a precheck function that runs against the planned submission list.

---

## Issue 3: F (MBP-1) volume two orders of magnitude over estimate

### What happened
F job submitted and accepted (cost $0.00, subscription-covered). Databento processed it and reported when done: **6,084,258,583 records, 486 GB compressed**. Started downloading, pulled 5.9 GB / 486 GB before I stopped the script.

### Why
The plan estimated "Total disk added: ~4-9 GB" for the full additional pull (trades + status + ED + F combined). For F alone, MBP-1 tick data for 4 symbols (ES, MES, NQ, MNQ) over 130 trading days is in fact 6.08 billion records / 486 GB — orders of magnitude bigger than the entire estimate.

### Root cause
MBP-1 (Market By Price Level 1) is a tick-by-tick L1 quote feed. Every quote update on the inside book is one record. For ES alone during US equity hours that's ~50k-200k updates per second; over 130 days you get a billion+ records per symbol. The estimate confused MBP-1 volume with a slower schema.

### Status
Stage 2 of the finisher stopped before pulling the rest. The 5.9 GB partial sits at `H:/Stock_Data/futures_dbn_staging/F/`. Databento retains completed batch jobs for ~30 days, so we can resume the download or drop it. Decision pending.

### What would have caught this
A volume-precheck via `metadata.get_record_count()` against the planned submission. The same precheck function from Issues 1-2 should also report record count and estimated bytes, surfacing scale issues before the user pays in time/disk.

---

## Common pattern across all three

All three issues are pre-submit oversight failures. The plan went `_submit(...)` immediately without:
1. Checking dollar cost (Issue 1)
2. Checking symbology resolution (Issue 2)
3. Checking record count / volume (Issue 3)

### Proposed fix for future Databento additions

Add a `precheck_section()` helper to `scripts/data/databento_batch_submit.py` that runs against the same kwargs as `_submit`:

```python
def precheck_section(client, section, schema, symbols, stype_in, start, end):
    # 1. Symbology
    try:
        res = client.symbology.resolve(dataset=DATASET, symbols=symbols,
                                        stype_in=stype_in, stype_out="instrument_id",
                                        start_date=start, end_date=end)
        n_resolved = sum(len(v) for v in res.get("result", {}).values())
        if n_resolved == 0:
            return f"FAIL: 0 symbols resolved"
    except Exception as e:
        return f"FAIL: symbology probe failed: {e}"

    # 2. Cost
    try:
        cost = client.metadata.get_cost(dataset=DATASET, schema=schema, symbols=symbols,
                                         stype_in=stype_in, start=start, end=end)
    except Exception as e:
        return f"FAIL: cost probe failed: {e}"

    # 3. Volume
    try:
        rc = client.metadata.get_record_count(dataset=DATASET, schema=schema,
                                               symbols=symbols, stype_in=stype_in,
                                               start=start, end=end)
        gb = client.metadata.get_billable_size(...) / 1e9
    except Exception:
        rc, gb = "?", "?"

    return f"OK: {n_resolved} syms, ${cost:.2f}, {rc:,} records, ~{gb:.1f} GB"
```

Then `submit_all` runs all prechecks first, prints a table, and bails or proceeds based on a `--confirm` flag with cost/size summary.

This converts all three issue classes from "fail-at-submit-time after partial commitment" to "fail-before-submit with a clear table to look at."
