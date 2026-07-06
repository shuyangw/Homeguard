from __future__ import annotations
from datetime import date
from src.data.artifacts import registry
from src.data.artifacts.currency_strength import CurrencyStrength
from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
from src.data.artifacts.pca_dollar import PcaDollar
from src.data.artifacts.spread_model import SpreadModel
from src.data.artifacts.vol_surface import VolSurface

# Register all builders as they are implemented (append in later phases).
registry.register(DailyOhlcCache())
registry.register(SpreadModel())
registry.register(VolSurface())
registry.register(CurrencyStrength())
registry.register(PcaDollar())


def list_components() -> list[dict]:
    out = []
    for name, b in registry.all_builders().items():
        out.append({
            "name": name,
            "kind": "artifact",
            "requires_key": getattr(b, "REQUIRES_KEY", None),
            "up_to_date": b.output_path().exists(),
        })
    return out


def build(names: list[str], start: date, end: date) -> None:
    order = registry.resolve_order(names)
    for n in order:
        registry.get_builder(n).build(start, end)
