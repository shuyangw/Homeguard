"""RAMP-CSP integration runner - wires RAMPSignals and MarketRegimeDetector into the CSP engine."""

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.settings.settings import PROJECT_ROOT, get_local_storage_dir, get_options_data_dir
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.ramp_strategy import RAMPSignals
from src.strategies.options.csp.contract_selector import CSPContractSelector
from src.strategies.options.csp.engine import CSPBacktestEngine, CSPBacktestResult
from src.strategies.options.data_loader import OptionsDataLoader
from src.utils.logger import get_logger

logger = get_logger()

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "strategies" / "ramp_csp.yaml"
SP500_LIST_PATH = PROJECT_ROOT / "config/universes" / "sp500-2025.csv"
REGIME_LOOKBACK_DAYS = 400


def load_csp_config(config_path: Optional[Path] = None) -> Dict:
    """Load RAMP-CSP YAML configuration.

    Args:
        config_path: Path to YAML file. Defaults to config/strategies/ramp_csp.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    logger.info(f"Loaded RAMP-CSP config from {config_path}")
    return config


def _load_sp500_symbols() -> List[str]:
    """Read S&P 500 symbols from the canonical backtest list CSV."""
    symbols: List[str] = []
    with open(SP500_LIST_PATH, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sym = row.get("Symbol", "").strip()
            if sym:
                symbols.append(sym)
    return symbols


class CSPBacktestRunner:
    """Runs a full RAMP-CSP backtest by wiring RAMPSignals into CSPBacktestEngine."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_csp_config()
        self._config = config

        options_dir = get_options_data_dir() / "options_combined"
        self._options_loader = OptionsDataLoader(options_dir)

        logger.info("CSPBacktestRunner initialized")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_options_symbols(self) -> List[str]:
        """Return symbols available in the options dataset."""
        return self._options_loader.get_available_symbols()

    def _load_equity_prices(
        self, symbols: List[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Load daily close prices from equities_daily_cache.parquet.

        Returns:
            DataFrame with DatetimeIndex (dates) and symbol columns containing close prices.
        """
        storage = get_local_storage_dir()
        cache_path = storage / "equities_daily_cache.parquet"

        if not cache_path.exists():
            logger.error(f"Equities daily cache not found at {cache_path}")
            return pd.DataFrame()

        logger.info(f"Loading equity prices from {cache_path}")
        df = pd.read_parquet(cache_path)

        # Filter to requested symbols
        symbols_set = set(symbols)
        df = df[df["symbol"].isin(symbols_set)].copy()

        # Parse dates and filter range
        df["trade_date"] = pd.to_datetime(df["trade_date"], utc=True)
        mask = (df["trade_date"].dt.date >= start_date) & (df["trade_date"].dt.date <= end_date)
        df = df.loc[mask]

        if df.empty:
            return pd.DataFrame()

        # Pivot to wide format: DatetimeIndex x symbol columns
        prices = df.pivot_table(index="trade_date", columns="symbol", values="close")
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        logger.info(f"Loaded equity prices: {len(prices)} days, {len(prices.columns)} symbols")
        return prices

    def _load_spy_vix(
        self, start_date: date, end_date: date
    ) -> Tuple[pd.Series, pd.Series]:
        """Load SPY and VIX close prices with extra lookback for regime detection.

        SPY comes from equities_daily_cache.parquet.
        VIX comes from options chain underlying_px (not in equities cache).

        Returns:
            (spy_series, vix_series) each indexed by DatetimeIndex.
        """
        lookback_start = start_date - timedelta(days=REGIME_LOOKBACK_DAYS)
        storage = get_local_storage_dir()

        # Load SPY from equities daily cache
        spy_series = pd.Series(dtype=float)
        cache_path = storage / "equities_daily_cache.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            spy_df = df[df["symbol"] == "SPY"].copy()
            spy_df["trade_date"] = pd.to_datetime(spy_df["trade_date"], utc=True)
            mask = (spy_df["trade_date"].dt.date >= lookback_start) & (spy_df["trade_date"].dt.date <= end_date)
            spy_df = spy_df.loc[mask].sort_values("trade_date")
            if not spy_df.empty:
                spy_series = pd.Series(
                    spy_df["close"].values,
                    index=pd.DatetimeIndex(spy_df["trade_date"]),
                )
                logger.info(f"SPY prices: {len(spy_series)} days")
        else:
            logger.warning(f"Equities cache not found at {cache_path}")

        # Load VIX from options chain underlying_px
        vix_series = self._load_underlying_px_series("VIX", lookback_start, end_date)
        logger.info(f"VIX prices: {len(vix_series)} days (from options chain)")

        return spy_series, vix_series

    def _load_underlying_px_series(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.Series:
        """Extract daily underlying_price from options chain EOD snapshots."""
        from datetime import time as dt_time

        eod_time = dt_time(16, 0)
        prices: Dict[date, float] = {}
        current = date(start_date.year, start_date.month, 1)
        end_month = date(end_date.year, end_date.month, 1)

        while current <= end_month:
            month_df = self._options_loader._load_month(
                symbol, current.year, current.month, target_time=eod_time
            )
            if not month_df.empty:
                for d in month_df["date"].unique():
                    if start_date <= d <= end_date:
                        day_rows = month_df[month_df["date"] == d]
                        prices[d] = float(day_rows["underlying_price"].iloc[0])

            # Move to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        if not prices:
            return pd.Series(dtype=float)

        sorted_dates = sorted(prices.keys())
        return pd.Series(
            [prices[d] for d in sorted_dates],
            index=pd.DatetimeIndex([datetime(d.year, d.month, d.day) for d in sorted_dates]),
        )

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None,
    ) -> CSPBacktestResult:
        """Execute a RAMP-CSP backtest over the given date range.

        Args:
            start_date: First trading day.
            end_date: Last trading day.
            symbols: Universe of equity symbols. Defaults to S&P 500 list.

        Returns:
            CSPBacktestResult with closed trades, daily snapshots, and equity curve.
        """
        if symbols is None:
            symbols = _load_sp500_symbols()
        logger.info(f"RAMP-CSP run: {start_date} -> {end_date}, {len(symbols)} symbols")

        # Load data (with lookback for RAMP momentum calculation)
        equity_lookback_start = start_date - timedelta(days=REGIME_LOOKBACK_DAYS)
        equity_prices = self._load_equity_prices(symbols, equity_lookback_start, end_date)
        if equity_prices.empty:
            logger.error("No equity price data loaded -- aborting")
            return CSPBacktestResult(closed_trades=[], daily_snapshots=[])

        spy_series, vix_series = self._load_spy_vix(start_date, end_date)
        if spy_series.empty or vix_series.empty:
            logger.error("SPY or VIX data missing -- aborting")
            return CSPBacktestResult(closed_trades=[], daily_snapshots=[])

        all_options = self._get_options_symbols()
        # Only keep symbols that are in the equity universe to avoid loading
        # massive index option chains (SPX ~16M rows/month) that cause OOM.
        equity_set = set(equity_prices.columns)
        options_symbols = [s for s in all_options if s in equity_set]
        logger.info(
            f"Options symbols filtered: {len(all_options)} available, "
            f"{len(options_symbols)} in equity universe"
        )

        # Build RAMP components
        ramp_cfg = self._config.get("ramp", {})
        available_syms = [s for s in symbols if s in equity_prices.columns]

        ramp_signals = RAMPSignals(
            symbols=available_syms,
            vix_threshold=ramp_cfg.get("vix_threshold", 25.0),
            spy_dd_threshold=ramp_cfg.get("spy_dd_threshold", -0.05),
        )

        regime_detector = MarketRegimeDetector()

        # ---- Callback definitions ----

        def get_regime(d: date) -> Tuple[str, float]:
            ts = datetime(d.year, d.month, d.day)
            spy_up_to = spy_series[spy_series.index.date <= d]
            vix_up_to = vix_series[vix_series.index.date <= d]
            if len(spy_up_to) < 200 or len(vix_up_to) < 252:
                return ("SIDEWAYS", 0.5)
            spy_df = pd.DataFrame({
                "close": spy_up_to,
                "open": spy_up_to,
                "high": spy_up_to,
                "low": spy_up_to,
                "volume": 1e6,
            })
            vix_df = pd.DataFrame({"close": vix_up_to})
            return regime_detector.classify_regime(spy_df, vix_df, ts)

        def get_crash_protection(d: date) -> bool:
            spy_up_to = spy_series[spy_series.index.date <= d]
            vix_up_to = vix_series[vix_series.index.date <= d]
            ramp_signals.update_historical_data(
                equity_prices[equity_prices.index.date <= d],
                spy_up_to,
                vix_up_to,
            )
            risk = ramp_signals.calculate_risk_signals(spy_up_to, vix_up_to)
            return risk.reduce_exposure

        def get_top_n_symbols(d: date) -> List[str]:
            prices_up_to = equity_prices[equity_prices.index.date <= d]
            spy_up_to = spy_series[spy_series.index.date <= d]
            vix_up_to = vix_series[vix_series.index.date <= d]
            ramp_signals.update_historical_data(prices_up_to, spy_up_to, vix_up_to)
            signals, _ = ramp_signals.generate_signals(
                current_positions={},
                prices_df=prices_up_to,
                spy_prices=spy_up_to,
                vix_prices=vix_up_to,
            )
            buy_signals = [s for s in signals if s.action == "buy"]
            buy_signals.sort(key=lambda s: s.rank)
            return [s.symbol for s in buy_signals]

        def get_chain(sym: str, d: date) -> pd.DataFrame:
            return self._options_loader.get_eod_chain(sym, d)

        def get_underlying_price(sym: str, d: date) -> float:
            if sym not in equity_prices.columns:
                return 0.0
            vals = equity_prices.loc[equity_prices.index.date <= d, sym].dropna()
            if vals.empty:
                return 0.0
            return float(vals.iloc[-1])

        # Build the engine
        strat_cfg = self._config.get("strategy", {})
        cs_cfg = self._config.get("contract_selection", {})

        selector = CSPContractSelector(
            target_delta_min=cs_cfg.get("target_delta_min", -0.35),
            target_delta_max=cs_cfg.get("target_delta_max", -0.25),
            min_dte=cs_cfg.get("min_dte", 21),
            max_dte=cs_cfg.get("max_dte", 35),
            min_open_interest=cs_cfg.get("min_open_interest", 100),
            max_spread_pct=cs_cfg.get("max_spread_pct", 0.15),
        )

        engine = CSPBacktestEngine(
            initial_capital=strat_cfg.get("initial_capital", 100_000),
            max_csp_allocation=strat_cfg.get("max_csp_allocation", 0.30),
            max_positions=strat_cfg.get("max_positions", 5),
            profit_target_pct=strat_cfg.get("profit_target_pct", 0.50),
            loss_limit_multiple=strat_cfg.get("loss_limit_multiple", 2.0),
            min_dte_exit=strat_cfg.get("min_dte_exit", 5),
            slippage_pct=strat_cfg.get("slippage_pct", 0.01),
            contract_fee=strat_cfg.get("contract_fee", 0.02),
            selector=selector,
        )

        # Determine trading days from the equity prices index
        trading_days = sorted(
            {d for d in equity_prices.index.date if start_date <= d <= end_date}
        )

        if not trading_days:
            logger.error("No trading days in range")
            return CSPBacktestResult(closed_trades=[], daily_snapshots=[])

        logger.info(f"Running CSP engine over {len(trading_days)} trading days")
        result = engine.run(
            trading_days=trading_days,
            get_regime=get_regime,
            get_crash_protection=get_crash_protection,
            get_top_n_symbols=get_top_n_symbols,
            get_chain=get_chain,
            get_underlying_price=get_underlying_price,
            options_symbols=options_symbols,
        )

        return result
