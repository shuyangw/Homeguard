"""Gate 0.3 committed driver for #36 equity-index inter-market RV (NQ/ES, RTY/ES).

Wraps `src.strategies.advanced.spread_intermarket_strategy.run_intermarket`
(already gate_return_stream-deflated per pair, persists returns.csv +
gate.json under output_dir) in RunStatus, appends one registry row per pair,
and writes one readiness report per pair. Was NQ/ES FAIL (0.329) / RTY/ES
REJECT (-0.280) pre-Gate-0.

CAVEAT-FIX (mandatory, per the retest TODO): runs the book-correlation check
vs the existing S&P equity-momentum sleeve (RAMP), which was never run for
this sleeve. If `--ramp-returns` is not supplied, book_corr is honestly
reported as NaN (not fabricated as 0/low). Consult
docs/methodology/backtesting.md Sections 1, 2, 3, 4, 6, 9, 12.
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.strategies.advanced.spread_intermarket_strategy import PAIRS, run_intermarket
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import append_registry_row, write_readiness_report

_STRATEGY_NAME = "FuturesIntermarketRV"


def _load_book_returns(path: str | None) -> pd.Series | None:
    """Load the equity-momentum (RAMP) daily return stream for the mandatory
    book-correlation check. Expects a CSV with columns date, return_pct (or
    return). Returns None (honest NaN downstream) if not supplied or unreadable."""
    if not path:
        return None
    try:
        df = pd.read_csv(path)
        date_col = "date" if "date" in df.columns else df.columns[0]
        ret_col = "return_pct" if "return_pct" in df.columns else (
            "return" if "return" in df.columns else df.columns[1])
        s = pd.Series(df[ret_col].to_numpy(dtype=float),
                       index=pd.to_datetime(df[date_col]))
        return s
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="#36 inter-market RV retest driver")
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS.keys()))
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--output-root", default="output/backtests/futures/intermarket")
    parser.add_argument("--report-root", default="docs/reports/futures")
    parser.add_argument("--ramp-returns", default=None,
                         help="CSV of the RAMP equity-momentum daily return stream "
                              "for the mandatory book-correlation check (Section 6). "
                              "If omitted, book_corr is reported as NaN, honestly.")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)
    book_returns = _load_book_returns(args.ramp_returns)

    for pair in args.pairs:
        out_dir = f"{args.output_root}/{pair}"
        with RunStatus(f"sp_retest_intermarket_{pair}", meta={"sleeve": "intermarket", "pair": pair}) as st:
            result = run_intermarket(pair, start_d, end_d, out_dir, book_returns=book_returns)
            st.heartbeat(note=f"gate computed: oos_sharpe={result.get('oos_sharpe')}")

        run_id = append_registry_row(f"{_STRATEGY_NAME}_{pair}", result,
                                      asset_class="futures", data_frequency="daily",
                                      universe_name=pair)

        corr = result.get("book_corr", float("nan"))
        corr_note = (
            "book_corr NOT computed -- pass --ramp-returns with the RAMP daily "
            "return stream to run the mandatory book-correlation check before "
            "trusting this sleeve's marginal value."
            if book_returns is None else
            f"book_corr={corr:.4f} vs the supplied equity-momentum book. "
            "A high positive correlation means re-expression / no marginal "
            "value regardless of standalone Sharpe (Section 6)."
        )
        extra = f"## Mandatory book-correlation check (Section 6)\n\n{corr_note}\n"
        write_readiness_report(f"{_STRATEGY_NAME}_{pair}", f"Inter-market RV {pair} (#36)",
                                result, f"{args.report_root}/INTERMARKET_{pair}_READINESS.md",
                                extra_notes=extra, run_id=run_id)
        print(f"[sp_retest_intermarket:{pair}] run_id={run_id} oos_sharpe={result.get('oos_sharpe')} "
              f"dsr={result.get('dsr')} pbo={result.get('pbo')} book_corr={corr}")


if __name__ == "__main__":
    main()
