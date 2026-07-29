"""Backfill the project-wide DSR trial counter in output/experiments.duckdb.

Context: `combinations_project` was never populated (max = 0) and
`combinations_in_run` was set on only 4 of 496 rows, so the cumulative
project-wide trial count that methodology Section 9.4 requires for the
Deflated Sharpe Ratio had to be reconstructed by hand. This script writes the
reconstruction back so it never has to be redone.

Derivation and every judgment call:
docs/strategies/research/options-slate/20260728_lifetime_trial_count.md

What it does, in one serialized write transaction (DuckDB is single-writer):

1. Appends one `verdict='tested'` row per OFF-LEDGER trial block that predates
   the registry (futures SP-A/B/C/E static baseline, RAMP equity v0-detector
   chain, RAMP options campaign, OpEx pinning). Each carries its block size in
   `combinations_in_run`, so `SUM(combinations_in_run)` becomes meaningful.
   These are aggregate provenance rows, not per-spec reconstructions -- the
   per-spec params were never recorded and are not inventable.

2. Sets `combinations_in_run` on every existing row: 1 for a distinct evaluated
   specification, 0 for an exact rerun of a spec already counted and for
   harness fixtures (ZeroForecastStub / ParamForecastStub).

3. Sets `combinations_project` on every row to the running cumulative count in
   `timestamp_utc` order, offset by the off-ledger blocks. Monotonic
   non-decreasing by construction -- N never shrinks.

Idempotent: re-running detects the sentinel off-ledger rows and stops.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

DB_PATH = REPO / "output" / "experiments.duckdb"
FIXTURE_STRATEGIES = {"ZeroForecastStub", "ParamForecastStub"}
SENTINEL_PREFIX = "offledger-"

# These blocks all predate the first registry row (2026-05-13), so they are
# timestamped before it: the cumulative counter must be monotonic in
# `timestamp_utc` order for a later reader to reproduce N as-of any date.
OFF_LEDGER_EPOCH = dt.datetime(2026, 5, 1)

# (run_id, strategy_name, asset_class, trials, window_start, window_end, note)
OFF_LEDGER_BLOCKS = [
    (
        f"{SENTINEL_PREFIX}futures-spabce-baseline",
        "FuturesCampaign-SP-A-B-C-E",
        "futures",
        40,
        "2010-06-07",
        "2026-02-20",
        "Static pre-registry futures baseline: SP-A 7 + SP-E 4 + SP-B 4 + "
        "SP-C 14 + pre-campaign carry/crypto sweep 11 = 40. Itemized at "
        "src/backtesting/walkforward_common.py CAMPAIGN_CUMULATIVE_TRIALS. "
        "29 of the 40 have a gradeable OOS Sharpe (CAMPAIGN_TRIAL_SHARPES); "
        "11 are ungradeable but were still evaluated, so they count toward N.",
    ),
    (
        f"{SENTINEL_PREFIX}ramp-equity-v0-detector-chain",
        "RAMP-v0-detector-family",
        "equity",
        36,
        "2017-01-01",
        "2026-05-16",
        "RAMP equity v0-detector trial chain, audited count 36 (V11+pre-V11 "
        "cohort 22 + V12+sensitivity 5 + V12c 1 + V13 1 + V14a/b/c 3 + V14a "
        "tau sens 2 + V14c dampen sens 2). Source docs/strategies/"
        "RAMP_VARIANTS.md. RECOVERED FROM A TRIAL-CHAIN RESET: the V20+ "
        "family restarted its counter at n_trials_project=1, zeroing these "
        "36. A lifetime N must not honor that reset.",
    ),
    (
        f"{SENTINEL_PREFIX}ramp-options-campaign",
        "RAMP-options-overlays",
        "options",
        16,
        "2018-07-01",
        "2025-12-31",
        "RAMP options overlay campaign closed 2026-04-02: 31 candidates "
        "catalogued, 16 distinct specs actually tested. Counted here at 16 "
        "distinct specs. NOT counted here: the 18-combination long-calls "
        "optimizer grid and the 75-combination ramp-csp grid (inner probes "
        "of two of the 16). Counting those grids raises this block to 109 -- "
        "the upper bound in the reconstruction doc.",
    ),
    (
        f"{SENTINEL_PREFIX}opex-pinning",
        "OpExPinning",
        "options",
        1,
        "2024-11-01",
        "2025-12-31",
        "Single OpEx pinning backtest on estimated gamma/OI (18 trades, "
        "11.1 pct win rate, -415 USD). One spec, no sweep. Source "
        "docs/strategies/20251230_OPEX_PINNING_STRATEGY_STATUS.md. Shelved.",
    ),
]


def spec_key(strategy, config_sha, notes, params, phase, window_start, window_end):
    """Identity of a distinct evaluated specification.

    Two disjoint writer conventions exist in the registry and neither alone
    is sufficient:
      - backtest_runner / robustness-runner rows carry a real `config_sha`
        and NULL `params` -> key on (strategy, config_sha, notes).
      - futures/fx harness rows carry `config_sha='unknown'` and a full
        `params` JSON -> key on (strategy, params, phase, window), with the
        bookkeeping fields stripped so a rerun under a different trial
        counter is not mistaken for a new spec.
    """
    if config_sha and config_sha != "unknown":
        return (strategy, config_sha, notes)
    if params:
        try:
            parsed = json.loads(params)
        except (TypeError, ValueError):
            parsed = {"raw": params}
        if isinstance(parsed, dict):
            parsed = {
                k: v for k, v in parsed.items()
                if k not in ("trial_count_project_wide", "dates")
            }
        return (strategy, json.dumps(parsed, sort_keys=True)[:2000],
                phase, str(window_start), str(window_end))
    return (strategy, phase, str(window_start), str(window_end))


def already_applied(con) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM runs WHERE run_id LIKE ?", [f"{SENTINEL_PREFIX}%"]
    ).fetchone()
    return bool(row and row[0])


def backfill(db_path: Path = DB_PATH) -> int:
    if not db_path.exists():
        raise FileNotFoundError(f"registry not found: {db_path}")

    backup = db_path.with_suffix(
        f".duckdb.bak-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(db_path, backup)
    logger.info(f"registry backed up to {backup}")

    con = duckdb.connect(str(db_path))
    try:
        if already_applied(con):
            logger.warning("off-ledger sentinel rows already present; "
                           "backfill already applied, nothing to do")
            return 0

        offset = sum(b[3] for b in OFF_LEDGER_BLOCKS)
        con.execute("BEGIN TRANSACTION")

        cumulative = 0
        for i, (run_id, strat, ac, trials, ws, we, note) in enumerate(OFF_LEDGER_BLOCKS):
            cumulative += trials
            con.execute(
                """INSERT INTO runs (run_id, timestamp_utc, strategy_name, agent_name,
                       phase, asset_class, window_start, window_end,
                       combinations_in_run, combinations_project, verdict, notes)
                   VALUES (?, ?, ?, 'trial-count-reconstruction', 'off_ledger_backfill',
                           ?, ?, ?, ?, ?, 'tested', ?)""",
                [run_id, OFF_LEDGER_EPOCH + dt.timedelta(seconds=i),
                 strat, ac, ws, we, trials, cumulative, note],
            )
        logger.info(f"inserted {len(OFF_LEDGER_BLOCKS)} off-ledger blocks "
                    f"totalling {offset} trials")

        rows = con.execute(
            """SELECT run_id, strategy_name, config_sha, notes, params, phase,
                      window_start, window_end
               FROM runs WHERE run_id NOT LIKE ?
               ORDER BY timestamp_utc, run_id""", [f"{SENTINEL_PREFIX}%"]
        ).fetchall()

        seen: set = set()
        running = offset
        updates = []
        n_new = n_rerun = n_fixture = 0
        for run_id, strat, csha, notes, params, phase, ws, we in rows:
            if strat in FIXTURE_STRATEGIES:
                in_run = 0
                n_fixture += 1
            else:
                key = spec_key(strat, csha, notes, params, phase, ws, we)
                if key in seen:
                    in_run = 0
                    n_rerun += 1
                else:
                    seen.add(key)
                    in_run = 1
                    n_new += 1
            running += in_run
            updates.append((in_run, running, run_id))

        con.executemany(
            "UPDATE runs SET combinations_in_run = ?, combinations_project = ? "
            "WHERE run_id = ?", updates)
        con.execute("COMMIT")

        logger.info(f"registry rows updated: {len(updates)} "
                    f"({n_new} distinct specs, {n_rerun} reruns, "
                    f"{n_fixture} harness fixtures)")
        logger.info(f"project-wide cumulative trial count N = {running} "
                    f"({offset} off-ledger + {n_new} registry)")
        return running
    except Exception:
        con.execute("ROLLBACK")
        logger.error("backfill failed and was rolled back", exc_info=True)
        raise
    finally:
        con.close()


if __name__ == "__main__":
    backfill()
