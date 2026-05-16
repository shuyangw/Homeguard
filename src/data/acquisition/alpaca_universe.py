"""Snapshot Alpaca's active, tradable US-equity universe."""

from pathlib import Path
from typing import Optional

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from src.utils.logger import get_logger

logger = get_logger(__name__)


def list_active_us_equities(
    trading_client: TradingClient,
    save_to: Optional[Path] = None,
) -> list[str]:
    """Snapshot Alpaca's active tradable US equities and ETFs.

    Filters: status=ACTIVE, asset_class=US_EQUITY, tradable=True. Drops symbols
    containing '/' or '.' (alt-ticker variants that lack historical bars).

    If save_to is set, persists the symbol list as a single-column CSV
    (column name 'Symbol') for audit and resume.
    """
    request = GetAssetsRequest(
        status=AssetStatus.ACTIVE,
        asset_class=AssetClass.US_EQUITY,
    )
    assets = trading_client.get_all_assets(request)

    symbols = sorted({
        a.symbol for a in assets
        if a.tradable
        and a.status == AssetStatus.ACTIVE
        and "/" not in a.symbol
        and "." not in a.symbol
    })

    logger.info(f"Alpaca universe snapshot: {len(symbols)} active US equities")

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Symbol": symbols}).to_csv(save_to, index=False)
        logger.info(f"Universe saved to {save_to}")

    return symbols
