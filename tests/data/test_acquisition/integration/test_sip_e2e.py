"""End-to-end integration test for the SIP redownload pipeline.

Requires:
  - .env populated with API_KEY/API_SECRET that has Algo Trader Plus access
  - Network connectivity
  - Approx 30 seconds wall-clock

Run with:
    pytest tests/data/test_acquisition/integration/test_sip_e2e.py -v -m integration
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from alpaca.data.enums import Adjustment, DataFeed

from src.data.acquisition.plugins.alpaca_equities import AlpacaEquitiesPlugin
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA
from src.data.acquisition.status_tracker import (
    rebuild_tracker,
    write_tracker_csv,
)


SYMBOLS = ["AAPL", "SPY", "MSFT", "NVDA", "TSLA"]
START = "2024-12-26"
END = "2024-12-31"


@pytest.mark.integration
class TestSipEndToEnd:
    def test_raw_pass_completes_and_tracker_is_good(self, tmp_path):
        plugin = AlpacaEquitiesPlugin(
            output_dir=tmp_path,
            feed=DataFeed.SIP,
            adjustment=Adjustment.RAW,
            storage_subdir_override="equities_1min_sip_raw",
            num_threads=3,
        )
        result = plugin.download(SYMBOLS, start_date=START, end_date=END)
        assert result.succeeded == len(SYMBOLS), result.failed_symbols

        # Verify folder structure matches existing pattern.
        for sym in SYMBOLS:
            sym_dir = tmp_path / "equities_1min_sip_raw" / f"symbol={sym}"
            assert sym_dir.exists()
            parquet_files = list(sym_dir.glob("year=*/month=*/data.parquet"))
            assert parquet_files, f"No parquet for {sym}"
            df = pd.read_parquet(parquet_files[0])
            assert list(df.columns) == CANONICAL_OHLCV_SCHEMA

        # Verify tracker
        tracker_rows = rebuild_tracker(
            base_dir=tmp_path,
            subdir="equities_1min_sip_raw",
            universe=SYMBOLS,
        )
        statuses = {r["symbol"]: r["status"] for r in tracker_rows}
        assert all(statuses[s] == "good" for s in SYMBOLS), statuses

    def test_resume_after_simulated_kill(self, tmp_path, monkeypatch):
        """Mark 2 symbols complete in manifest, run, verify only 3 are fetched."""
        plugin = AlpacaEquitiesPlugin(
            output_dir=tmp_path,
            feed=DataFeed.SIP,
            adjustment=Adjustment.RAW,
            storage_subdir_override="equities_1min_sip_raw",
            num_threads=3,
        )
        # Pre-mark 2 symbols as already complete
        for sym in SYMBOLS[:2]:
            plugin.manifest.set_entry(sym, status="complete", rows=999)
        plugin.manifest.save()

        # Filter the universe to those not complete
        existing = {
            s for s, e in plugin.manifest.get_all_entries().items()
            if e.get("status") == "complete"
        }
        remaining = [s for s in SYMBOLS if s not in existing]
        assert len(remaining) == 3

        result = plugin.download(remaining, start_date=START, end_date=END)
        assert result.total_symbols == 3
        assert result.succeeded == 3

    def test_raw_vs_split_consistency_smoke(self, tmp_path):
        for adj_name, adjustment, subdir in [
            ("raw", Adjustment.RAW, "equities_1min_sip_raw"),
            ("split", Adjustment.SPLIT, "equities_1min_sip_split"),
        ]:
            plugin = AlpacaEquitiesPlugin(
                output_dir=tmp_path,
                feed=DataFeed.SIP,
                adjustment=adjustment,
                storage_subdir_override=subdir,
                num_threads=3,
            )
            plugin.download(SYMBOLS[:2], start_date=START, end_date=END)

        for sym in SYMBOLS[:2]:
            raw_paths = sorted((
                tmp_path / "equities_1min_sip_raw" / f"symbol={sym}"
            ).glob("year=*/month=*/data.parquet"))
            split_paths = sorted((
                tmp_path / "equities_1min_sip_split" / f"symbol={sym}"
            ).glob("year=*/month=*/data.parquet"))
            assert len(raw_paths) == len(split_paths)
            for rp, sp in zip(raw_paths, split_paths):
                raw_df = pd.read_parquet(rp, columns=["timestamp", "trade_count"])
                split_df = pd.read_parquet(sp, columns=["timestamp", "trade_count"])
                assert raw_df["timestamp"].equals(split_df["timestamp"])
                assert raw_df["trade_count"].equals(split_df["trade_count"])
