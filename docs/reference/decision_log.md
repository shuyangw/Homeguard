# Decision Log Reference

## Purpose

One JSON record per strategy trigger fire. Captures inputs, preconditions,
logic decisions, executions, post-state, and error (if any). Designed to
make "why did/didn't strategy X trade Y?" a single file lookup.

## Storage

- **One JSONL per strategy per date:** `data/trading/decisions/<strategy>_<YYYYMMDD>.jsonl`
- **Latest snapshot per strategy:** `data/trading/decisions/_latest/<strategy>.json`
- **Retention:** 1 year, enforced lazily by writer
- **Schema:** `src/trading/decision_log/record.py`

## CLI quickstart

```bash
# What did RAMP just decide?
python -m src.trading.decision_log show ramp

# Last 7 days of RAMP decisions, table format
python -m src.trading.decision_log list ramp --days 7

# Why didn't RAMP trade AVAX on 04-20?
python -m src.trading.decision_log explain ramp --symbol AVAX --date 2026-04-20

# All RAMP decisions where 0 orders went through, last 30 days
python -m src.trading.decision_log grep ramp --where 'executions:length=0' --days 30

# Cross-strategy weekly roll-up
python -m src.trading.decision_log summary --days 7
```

## Filter DSL (for `grep --where`)

| Expression | Meaning |
|---|---|
| `regime=BEAR` | `inputs.regime == 'BEAR'` |
| `executions:length=0` | `len(rec.executions) == 0` |
| `error.stage=logic` | `rec.error.stage == 'logic'` |
| `preconditions.all_passed=false` | `rec.preconditions.all_passed is False` |

Extend by editing `_compile_predicate` in `src/trading/decision_log/reader.py`.

## Investigation runbook

### "Why didn't strategy X trade today?"

```bash
python -m src.trading.decision_log show <strategy>
```

Look at:
1. **Preconditions** -- any gate failing? (health_check, data_freshness, lock)
2. **Inputs** -- was data available? Regime detected? Universe size matches expected?
3. **Logic** -- was target_symbols empty? (regime -> reduce_exposure?)
4. **Executions** -- entries with `status: skipped` or `status: rejected`? Read the `reason` field.
5. **Error** -- if non-null, the `stage` field tells you where it died.

### "Strategy generated 10 signals but placed 0 orders"

```bash
python -m src.trading.decision_log show <strategy> --json | jq '.executions[] | {symbol, status, reason}'
```

The `reason` per execution is the per-symbol rejection cause.

### "What was strategy X's regime distribution last month?"

```python
from datetime import date
from src.trading.decision_log.reader import iter_records
from collections import Counter

records = iter_records('ramp', since=date(2026, 3, 24), until=date(2026, 4, 23))
print(Counter(r.inputs.regime for r in records))
```

## Schema reference

See `docs/superpowers/specs/2026-04-24-decision-log-observability-design.md`
section "Schema" for the full dataclass definitions.

`schema_version` is the contract field -- readers branch on it. Bump when
fields are added or removed; existing records with old versions are still
parsable (extra fields are added with defaults; removed fields are tolerated).

## Adding a new strategy

A class-based strategy adapter (subclass of `StrategyAdapter`) gets the
helpers automatically. In `run_once`, follow the staged pattern:

```python
def run_once(self):
    rec = self._begin_decision('scheduled_rebalance', schedule_time='15:55')
    try:
        with self._stage(rec, 'preconditions'):
            if not self._check_common_preconditions(rec):
                return
            # populate strategy-specific gates
        with self._stage(rec, 'inputs'):
            rec.inputs = self._build_decision_inputs()
        with self._stage(rec, 'logic'):
            rec.logic_decisions = self._build_decision_logic(...)
        with self._stage(rec, 'execution'):
            self._execute(...)  # mutates rec.executions
        with self._stage(rec, 'post_state'):
            rec.post_state = self._snapshot_post_state()
    except Exception as e:
        rec.error = ErrorInfo(...)
        raise
    finally:
        self._write_decision(rec)
```

Module-style strategies (CSCM-shaped) call `decision_log.begin_decision`,
`decision_log.stage`, `decision_log.write_decision` directly.

## What it doesn't replace

- `trades.csv` (compliance audit; flat row per fill)
- journald -> Loki (systemd default; complementary low-cardinality stream)
- `strategy_positions.json` (state file with locks; not a log)
- `market_checks.csv` (15s heartbeat; not per-trigger)
