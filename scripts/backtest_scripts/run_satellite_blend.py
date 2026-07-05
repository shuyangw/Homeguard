"""Core-satellite blended walk-forward gate runner (crypto satellite blend Task 3).

Runs the carry (core) walk-forward and the crypto (satellite) walk-forward
independently via `walk_forward_carver(..., return_window_returns=True)`,
then blends their per-window dated OOS return streams via
`src/backtesting/blend/satellite_blend.py::blend_books` and reports the
blended statistical gate (Sharpe / PBO / PSR / DSR) alongside the two
standalone book gates.

Usage:
    python -m scripts.backtest_scripts.run_satellite_blend \
        --core-config config/backtesting/futures/carver_tsmom.yaml \
        --sat-config config/backtesting/crypto/<crypto_config>.yaml \
        --sat-weight 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.backtest_scripts.run_carver_walkforward import (
    _config_to_kwargs,
    walk_forward_carver,
)
from src.backtesting.blend.satellite_blend import blend_books
from src.utils.logger import get_logger

_WALK_FORWARD_TRAIN_MONTHS = 36
_WALK_FORWARD_TEST_MONTHS = 12
_WALK_FORWARD_STEP_MONTHS = 12


def _run_wf(config_path: str, max_workers: int | None) -> dict:
    """Load a futures/crypto backtest YAML config and run its walk-forward gate."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    kw = _config_to_kwargs(cfg)
    return walk_forward_carver(
        train_months=_WALK_FORWARD_TRAIN_MONTHS,
        test_months=_WALK_FORWARD_TEST_MONTHS,
        step_months=_WALK_FORWARD_STEP_MONTHS,
        start=kw["start"],
        end=kw["end"],
        universe=kw["universe"],
        capital=kw["capital"],
        vol_target=kw["vol_target"],
        strategy_name=kw["strategy_name"],
        strategy_params=kw.get("strategy_params", {}),
        idm=kw.get("idm", False),
        idm_cap=kw.get("idm_cap"),
        max_workers=max_workers,
        return_window_returns=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend a core (carry) and satellite (crypto) walk-forward "
        "OOS return stream and report the blended statistical gate."
    )
    parser.add_argument("--core-config", required=True, help="Path to the core (carry) futures backtest YAML config")
    parser.add_argument("--sat-config", required=True, help="Path to the satellite (crypto) backtest YAML config")
    parser.add_argument("--sat-weight", type=float, default=0.15, help="Satellite book weight in the blend (0-1)")
    parser.add_argument("--jobs", type=int, default=None, help="Max worker processes for each walk-forward run")
    parser.add_argument("--json", default=None, help="Optional path to write the blended gate result as JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log = get_logger(__name__)

    log.info(f"Running core (carry) walk-forward: {args.core_config}")
    core = _run_wf(args.core_config, args.jobs)
    log.info(f"Running satellite (crypto) walk-forward: {args.sat_config}")
    sat = _run_wf(args.sat_config, args.jobs)

    core_windows = core["per_window_oos"]
    # Concat satellite dated OOS returns into ONE series; blend_books reindexes
    # it onto each core window's dates (missing -> 0), so a schedule mismatch
    # between the core and satellite walk-forward windows is handled cleanly.
    sat_all = pd.concat(sat["per_window_oos"]).sort_index()
    sat_windows = [sat_all for _ in core_windows]

    blended = blend_books(core_windows, sat_windows, sat_weight=args.sat_weight)

    out = {
        "core_sharpe": core["oos_sharpe"],
        "core_pbo": core["pbo"],
        "sat_sharpe": sat["oos_sharpe"],
        "sat_pbo": sat["pbo"],
        "sat_weight": args.sat_weight,
        "blended_sharpe": blended["oos_sharpe"],
        "blended_pbo": blended["pbo"],
        "blended_psr": blended["psr"],
        "blended_dsr": blended["dsr"],
        "blended_skew": blended["skew"],
        "blended_kurt": blended["kurtosis_pearson"],
        "n_oos_days": blended["n_oos_days"],
    }
    log.info("SATELLITE_BLEND " + json.dumps(out))
    log.info(
        f"core carry {out['core_sharpe']:.4f}/PBO{out['core_pbo']:.4f} | "
        f"crypto {out['sat_sharpe']:.4f}/PBO{out['sat_pbo']:.4f} | "
        f"BLEND@{args.sat_weight:.0%} {out['blended_sharpe']:.4f}/PBO{out['blended_pbo']:.4f}"
    )

    if args.json:
        Path(args.json).write_text(json.dumps(out))
        log.info(f"Wrote blended gate result to {args.json}")


if __name__ == "__main__":
    main()
