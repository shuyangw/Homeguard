"""#28 VRP signal expressed as a VRP-sized short-VX1 stream, gated + checked for
re-expression against #26. VRP = ATM implied vol - HAR realized-vol forecast."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir
from src.backtesting.vol.atm_iv import atm_iv_series
from src.backtesting.vol.har_rv import har_forecast_vol_annualized
from src.backtesting.vix.vix_rolldown_eval import rolldown_returns
from src.backtesting.walkforward_common import gate_return_stream, _annualized_sharpe
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TOP_BAND = 0.5  # short-vol only when VRP percentile is above this (pre-registered)


def percentile_rank_causal(s: pd.Series, window: int = 252) -> pd.Series:
    """Rank each value against the `window` strictly-prior values (excludes itself).

    Uses a rolling window of `window + 1` so the trailing `window` entries feeding
    the rank are all before the current observation; the first `window` outputs
    are NaN (not enough prior history yet).
    """
    def _rank(x):
        return (x[:-1] < x[-1]).mean()
    return s.rolling(window + 1).apply(_rank, raw=True).rename("vrp_pct")


def vrp_signal(root: str, start: date, end: date, window: int = 252) -> pd.Series:
    iv = atm_iv_series(root, start, end)
    hv = har_forecast_vol_annualized(root, start, end)
    vrp = (iv - hv.reindex(iv.index)).dropna().rename("vrp")
    return percentile_rank_causal(vrp, window=window)


def _vx1_daily_pnl() -> pd.Series:
    curve = pl.read_parquet(get_local_storage_dir() / "alt_data" / "vix" / "vx_curve.parquet").to_pandas()
    curve["date"] = pd.to_datetime(curve["date"])
    c = curve.sort_values("date").set_index("date")
    ret = c["vx1_settle"].pct_change(fill_method=None)
    if "vx1_dte" in c.columns:
        ret = ret.mask(c["vx1_dte"].diff() > 0, 0.0)  # roll mask, same as #26
    return ret.rename("vx1_ret")


def vrp_return_stream(root: str, start: date, end: date) -> pd.Series:
    pct = vrp_signal(root, start, end)
    vx1_ret = _vx1_daily_pnl()
    # position: short VX1 sized by prior-day VRP percentile when above the top band
    sized = pct.where(pct >= _TOP_BAND, 0.0)
    position = (-1.0 * sized).shift(1)
    stream = (position * vx1_ret.reindex(position.index)).dropna()
    return stream.rename("vrp_return")


def reexpression_stats(stream: pd.Series, ref: pd.Series) -> dict:
    j = pd.concat([stream.rename("a"), ref.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 30:
        return {"corr": float("nan"), "marginal_sharpe": float("nan")}
    corr = float(j["a"].corr(j["b"]))
    # marginal Sharpe of a over b: Sharpe of the residual after regressing a on b
    beta = np.cov(j["a"], j["b"])[0, 1] / np.var(j["b"])
    resid = j["a"] - beta * j["b"]
    return {"corr": corr, "marginal_sharpe": _annualized_sharpe(resid.values)}


def run_vrp(root: str, start: date, end: date, output_dir) -> dict:
    stream = vrp_return_stream(root, start, end)
    stream.index = pd.to_datetime(stream.index)
    result = gate_return_stream(stream)
    ref = rolldown_returns(pl.read_parquet(
        get_local_storage_dir() / "alt_data" / "vix" / "vx_curve.parquet").to_pandas())
    ref.index = pd.to_datetime(ref.index)
    result.update({f"reexpr_{k}": v for k, v in reexpression_stats(stream, ref).items()})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stream.to_frame("return").to_csv(out / "returns.csv", index_label="date")
    (out / "gate.json").write_text(json.dumps(result, default=float, indent=2))
    logger.info(f"[vrp:{root}] {result}")
    return result
