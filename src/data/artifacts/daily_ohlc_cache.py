from __future__ import annotations
from datetime import date
from pathlib import Path
from src.data.artifacts.base import ArtifactBuilder
from src.settings import get_local_storage_dir
from scripts.data.build_fx_daily_cache import build_fx_daily_cache

G10_PAIRS = [
    "GBPUSD", "USDCAD", "AUDUSD", "NZDUSD", "AUDNZD", "AUDJPY", "NZDJPY",
    "EURNOK", "EURSEK", "USDNOK", "USDSEK", "NOKSEK", "NOKJPY", "SEKJPY",
]
EXISTING_PAIRS = [
    "EURUSD", "USDJPY", "USDCHF", "EURJPY", "EURCHF", "CHFJPY", "XAUUSD", "XAGUSD",
]


class DailyOhlcCache(ArtifactBuilder):
    name = "daily_ohlc_cache"
    output_subdir = "daily_ohlc_cache"

    def inputs(self) -> list[str]:
        return ["minute"]

    def target_pairs(self) -> list[str]:
        return EXISTING_PAIRS + G10_PAIRS

    def output_path(self) -> Path:
        return get_local_storage_dir() / "fx_daily"

    def build(self, start: date, end: date) -> Path:
        build_fx_daily_cache(self.target_pairs(), start, end)
        return self.output_path()
