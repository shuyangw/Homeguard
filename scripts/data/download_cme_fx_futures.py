"""Bulk-pull CME FX futures per config/universes/cme_fx_futures-2026.csv.

Tier 1 contracts: OHLCV-1m AND MBP-1
Tier 2 contracts: OHLCV-1m only (E-micros)
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.databento_futures import DatabentoFuturesPlugin
from src.utils.logger import get_logger
logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path,
                        default=PROJECT_ROOT / "config" / "universes" / "cme_fx_futures-2026.csv")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--schemas", nargs="+", default=["ohlcv-1m", "mbp-1"],
                        help="Subset: ohlcv-1m, mbp-1, or both")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.universe.open(encoding="utf-8")))
    plugin = DatabentoFuturesPlugin()

    # Group by tier so we can pass the right schema set
    tier1_symbols = [r["symbol"] for r in rows if int(r["tier"]) == 1]
    tier2_symbols = [r["symbol"] for r in rows if int(r["tier"]) == 2]

    logger.info(f"Tier 1 symbols ({len(tier1_symbols)}): ohlcv-1m + mbp-1 (if in --schemas)")
    logger.info(f"Tier 2 symbols ({len(tier2_symbols)}): ohlcv-1m only")

    # NOTE: actual download invocation depends on DatabentoFuturesPlugin's public API.
    # The existing entry point is:
    #   plugin.download(symbols, start_date, end_date, skip_existing)
    # where `plugin` is constructed with schema="ohlcv-1m" or schema="mbp-1".
    # For a two-schema pull, instantiate once per schema:
    #
    #   ohlcv_plugin = DatabentoFuturesPlugin(schema="ohlcv-1m")
    #   ohlcv_plugin.download(tier1_symbols + tier2_symbols, "2010-06-06",
    #                         args.end.isoformat(), skip_existing=True)
    #
    #   if "mbp-1" in args.schemas:
    #       mbp1_plugin = DatabentoFuturesPlugin(schema="mbp-1")
    #       mbp1_plugin.download(tier1_symbols, "2010-06-06",
    #                            args.end.isoformat(), skip_existing=True)
    #
    # However, _fetch_symbol_data currently routes mbp-1 to the same trades path
    # (_normalize_trades) and _get_storage_subdir returns "futures_trades" for any
    # non-ohlcv-1m schema. Before executing a real mbp-1 pull, extend those two
    # methods to handle mbp-1 (route to _write_mbp1_partition instead of pandas
    # parquet). That wiring is the remaining follow-up item.

    raise NotImplementedError(
        "Bulk-pull invocation TBD -- the plugin's download() -> _fetch_symbol_data() "
        "pipeline does not yet route mbp-1 through _write_mbp1_partition. "
        "Extend _fetch_symbol_data and _get_storage_subdir for mbp-1, then replace "
        "this stub with the two-plugin loop shown in the comments above."
    )


if __name__ == "__main__":
    raise SystemExit(main())
