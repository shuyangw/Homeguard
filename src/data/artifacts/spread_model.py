from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.settings import get_local_storage_dir
from src.utils import logger

_METALS = {"XAU", "XAG"}
TIER_BASE_PIPS = {"major": 1.0, "minor": 3.0, "metal": 3.0}
ROLLOVER_HOUR_UTC = 21
ROLLOVER_MULT = 5.0


def _tier(pair: str) -> str:
    if pair[:3] in _METALS or pair[3:] in _METALS:
        return "metal"
    if "USD" in (pair[:3], pair[3:]):
        return "major"
    return "minor"


def _pip_size(pair: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise (matches src/backtesting/costs/fx.py)."""
    return 0.01 if pair[3:] == "JPY" else 0.0001


def synthetic_spread(pair: str, hour_of_week: int, anchors: dict[str, float]) -> float:
    base = anchors[pair] if pair in anchors else TIER_BASE_PIPS[_tier(pair)]
    hour_utc = hour_of_week % 24
    mult = ROLLOVER_MULT if hour_utc == ROLLOVER_HOUR_UTC else 1.0
    return base * mult


class SpreadModel(ArtifactBuilder):
    name = "spread_model"
    output_subdir = "spread_model"

    def inputs(self) -> list[str]:
        return ["quotes", "minute"]

    def _quote_pairs(self) -> list[str]:
        root = get_local_storage_dir() / "fx" / "massive" / "quotes_minute_aggregated"
        if not root.exists():
            return []
        return [p.name.replace("symbol=", "") for p in root.glob("symbol=*")]

    def _anchor_for_pair(self, pair: str) -> float | None:
        sym_dir = get_local_storage_dir() / "fx" / "massive" / "quotes_minute_aggregated" / f"symbol={pair}"
        files = list(sym_dir.glob("**/*.parquet"))
        if not files:
            return None
        df = pl.scan_parquet([str(f) for f in files]).select("spread_p50").collect()
        if df.is_empty():
            return None
        pip_size = _pip_size(pair)
        return float(df["spread_p50"].median()) / pip_size

    def build(self, start: date, end: date) -> Path:
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        anchors: dict[str, float] = {}
        for pair in self._quote_pairs():
            anchor = self._anchor_for_pair(pair)
            if anchor is not None:
                anchors[pair] = anchor
        rows = []
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        for pair in DailyOhlcCache().target_pairs():
            for how in range(168):
                rows.append({"pair": pair, "hour_of_week": how,
                             "spread_pips": synthetic_spread(pair, how, anchors)})
        table = pl.DataFrame(rows)
        tmp = out_dir / "table.parquet.tmp"
        table.write_parquet(tmp)
        os.replace(tmp, out_dir / "table.parquet")
        logger.info(f"[spread_model] wrote {len(rows)} rows, {len(anchors)} anchors")
        return out_dir
