"""
OpEx Strategy Backtest Runner.

Specialized backtest engine for OpEx Pinning strategy that combines:
1. OHLCV price data from standard data sources
2. Options chain data from ThetaData
3. Pre-computed GEX summaries for signal generation

Usage:
    runner = OpExBacktestRunner(
        initial_capital=100_000,
        symbols=['SPY', 'QQQ', 'IWM'],
    )

    # Run backtest with pre-computed GEX
    results = runner.run_backtest(
        start_date='2020-01-01',
        end_date='2024-12-31',
        gex_data=gex_summaries,
    )

    # Analyze results
    metrics = runner.calculate_metrics(results)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.strategies.opex.calendar import OpExCalendar, OpExPhase
from src.strategies.opex.gex_calculator import GEXCalculator
from src.strategies.opex.signal_generator import (
    OpExSignalGenerator,
    SignalConfig,
    SignalDirection,
    OpExSignal,
)
from src.strategies.advanced.opex_pinning_strategy import OpExPinningStrategy
from src.data.options.options_schema import DailyGEXSummary, OptionsChain
from src.data.options.options_store import OptionsDataStore
from src.data.options.thetadata_adapter import ThetaDataAdapter
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader
from src.utils.logger import logger


@dataclass
class OpExTradeRecord:
    """Record of a single OpEx trade."""
    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    direction: SignalDirection
    phase: OpExPhase
    pin_strike: float
    distance_to_pin: float
    net_gex: float
    pnl: float
    pnl_pct: float
    shares: int


@dataclass
class OpExBacktestResults:
    """Results from OpEx strategy backtest."""
    trades: List[OpExTradeRecord] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None

    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Phase breakdown
    pinning_trades: int = 0
    pinning_win_rate: float = 0.0
    post_opex_trades: int = 0
    post_opex_win_rate: float = 0.0


class OpExBacktestRunner:
    """
    Specialized backtest runner for OpEx Pinning strategy.

    Handles the unique requirements of OpEx backtesting:
    - Integration of OHLCV and options data
    - GEX-based signal generation
    - Phase-specific analysis
    - Walk-forward validation support
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        symbols: Optional[List[str]] = None,
        position_size_pct: float = 0.10,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        strategy_params: Optional[Dict] = None,
    ):
        """
        Initialize OpEx backtest runner.

        Args:
            initial_capital: Starting capital
            symbols: Symbols to trade (default: SPY, QQQ, IWM)
            position_size_pct: Position size as fraction of capital
            commission_pct: Commission as percentage
            slippage_pct: Slippage as percentage
            strategy_params: Additional strategy parameters
        """
        self.initial_capital = initial_capital
        self.symbols = symbols or ['SPY', 'QQQ', 'IWM']
        self.position_size_pct = position_size_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # Strategy parameters
        default_params = {
            'min_distance_pct': 0.005,
            'max_distance_pct': 0.03,
            'active_phases': ['PINNING', 'POST_OPEX'],
            'holding_period': 'overnight',
            'position_size_pct': position_size_pct,
        }
        if strategy_params:
            default_params.update(strategy_params)

        self.strategy = OpExPinningStrategy(params=default_params)
        self.calendar = OpExCalendar()
        self.gex_calc = GEXCalculator()

        # Data loaders - use 1min data and resample to daily
        self.price_loader = StreamingDataLoader(timeframe='1min')
        self.options_store = OptionsDataStore()
        self.options_adapter = ThetaDataAdapter()

        logger.info("Initialized OpEx Backtest Runner")
        logger.info(f"  Capital: ${initial_capital:,.0f}")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Position Size: {position_size_pct:.1%}")

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        gex_data: Optional[Dict[str, List[DailyGEXSummary]]] = None,
        use_stored_gex: bool = True,
        compute_from_options: bool = False,
    ) -> OpExBacktestResults:
        """
        Run backtest for the specified period.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            gex_data: Pre-computed GEX data by symbol
            use_stored_gex: If True, load GEX from options store
            compute_from_options: If True, compute GEX from raw ThetaData parquet files

        Returns:
            OpExBacktestResults with trades and metrics
        """
        logger.info(f"\nRunning OpEx backtest: {start_date} to {end_date}")

        # Load price data
        price_data = self._load_price_data(start_date, end_date)
        if price_data.empty:
            logger.warning("No price data loaded")
            return OpExBacktestResults()

        # Load or use provided GEX data
        if gex_data is None:
            if compute_from_options:
                logger.info("Computing GEX from raw options data...")
                gex_data = self._compute_gex_from_options(start_date, end_date)
            elif use_stored_gex:
                gex_data = self._load_gex_data(start_date, end_date)

        # Run simulation
        results = self._simulate_trading(price_data, gex_data)

        # Calculate metrics
        self._calculate_metrics(results)

        return results

    def _load_price_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load OHLCV price data for all symbols and resample to daily."""
        all_data = []

        for symbol in self.symbols:
            try:
                for sym, df in self.price_loader.iter_symbols([symbol], start_date, end_date):
                    pdf = df.to_pandas()
                    pdf['symbol'] = symbol
                    all_data.append(pdf)
            except Exception as e:
                logger.warning(f"Failed to load price data for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.set_index('timestamp').sort_index()

        # Resample to daily OHLCV
        daily_data = []
        for symbol in self.symbols:
            symbol_data = combined[combined['symbol'] == symbol].copy()
            if symbol_data.empty:
                continue

            # Resample to daily
            daily = symbol_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna()

            daily['symbol'] = symbol
            daily_data.append(daily)

        if not daily_data:
            return pd.DataFrame()

        result = pd.concat(daily_data)
        logger.info(f"Loaded {len(result)} daily price records for {len(self.symbols)} symbols")
        return result

    def _load_gex_data(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, List[DailyGEXSummary]]:
        """Load pre-computed GEX data from options store."""
        gex_data = {}

        for symbol in self.symbols:
            try:
                summaries = self.options_store.get_gex_summaries(
                    symbol=symbol,
                    start_date=date.fromisoformat(start_date),
                    end_date=date.fromisoformat(end_date),
                )
                if summaries:
                    gex_data[symbol] = summaries
                    logger.info(f"Loaded {len(summaries)} GEX summaries for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load GEX data for {symbol}: {e}")

        return gex_data

    def _compute_gex_from_options(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, List[DailyGEXSummary]]:
        """
        Compute GEX data from raw ThetaData parquet files.

        Reads options chains, calculates GEX, and returns summaries.
        """
        gex_data = {}
        start_d = date.fromisoformat(start_date)
        end_d = date.fromisoformat(end_date)

        for symbol in self.symbols:
            summaries = []
            trading_dates = self.options_adapter.get_trading_dates(symbol, start_d, end_d)

            if not trading_dates:
                logger.warning(f"No options data for {symbol}")
                continue

            logger.info(f"Computing GEX for {symbol}: {len(trading_dates)} trading days")

            for d in trading_dates:
                try:
                    chain = self.options_adapter.load_eod_chain(symbol, d)
                    if chain is None or not chain.contracts:
                        continue

                    # Calculate GEX
                    total_gex = self.gex_calc.calculate_total_gex(chain)
                    strike_gex = self.gex_calc.calculate_strike_gex(chain)
                    pin_strike = self.gex_calc.find_pin_strike(chain, method='max_gex')

                    if pin_strike is None:
                        pin_strike = chain.underlying_price

                    # Calculate component GEX
                    call_gex = sum(
                        self.gex_calc.calculate_contract_gex(
                            c.gamma, c.open_interest, chain.underlying_price, is_call=True
                        )
                        for c in chain.get_calls()
                    )
                    put_gex = sum(
                        self.gex_calc.calculate_contract_gex(
                            c.gamma, c.open_interest, chain.underlying_price, is_call=False
                        )
                        for c in chain.get_puts()
                    )

                    # Distance to pin
                    distance = (chain.underlying_price - pin_strike) / pin_strike if pin_strike else 0

                    # Get GEX at pin strike
                    pin_gex_obj = strike_gex.get(pin_strike) if strike_gex else None
                    pin_gex_value = pin_gex_obj.net_gex if pin_gex_obj else 0.0

                    summary = DailyGEXSummary(
                        symbol=symbol,
                        date=d,
                        expiry=chain.expiry,
                        underlying_price=chain.underlying_price,
                        total_gex=total_gex,
                        call_gex=call_gex,
                        put_gex=put_gex,
                        pin_strike=pin_strike,
                        pin_strike_gex=pin_gex_value,
                        distance_to_pin=distance,
                        total_oi=chain.total_oi,
                        put_call_ratio=chain.put_call_ratio,
                    )
                    summaries.append(summary)

                except Exception as e:
                    logger.debug(f"Failed to compute GEX for {symbol} on {d}: {e}")
                    continue

            if summaries:
                gex_data[symbol] = summaries
                logger.info(f"Computed {len(summaries)} GEX summaries for {symbol}")

        return gex_data

    def _simulate_trading(
        self,
        price_data: pd.DataFrame,
        gex_data: Optional[Dict[str, List[DailyGEXSummary]]],
    ) -> OpExBacktestResults:
        """Simulate trading based on OpEx signals."""
        # Handle empty data
        if price_data.empty:
            return OpExBacktestResults()

        trades: List[OpExTradeRecord] = []
        equity = [self.initial_capital]
        equity_dates = [price_data.index[0].date()]

        # Convert GEX data to lookup dict
        gex_lookup: Dict[Tuple[str, date], DailyGEXSummary] = {}
        if gex_data:
            for symbol, summaries in gex_data.items():
                for summary in summaries:
                    gex_lookup[(symbol, summary.date)] = summary

        # Track positions
        positions: Dict[str, Dict] = {}  # symbol -> position info
        capital = self.initial_capital

        # Get unique dates
        dates = price_data.index.normalize().unique().sort_values()

        for current_dt in dates:
            current_date = current_dt.date()

            # Get phase
            phase = self.calendar.get_phase(current_date)

            # Process each symbol
            for symbol in self.symbols:
                # Get price data for this symbol and date
                symbol_data = price_data[price_data['symbol'] == symbol]
                day_data = symbol_data[symbol_data.index.date == current_date]

                if len(day_data) == 0:
                    continue

                close_price = day_data['close'].iloc[-1]
                open_price = day_data['open'].iloc[0] if 'open' in day_data else close_price

                # Check for exit first
                if symbol in positions:
                    pos = positions[symbol]

                    # Exit condition: next day open for overnight hold
                    if pos['entry_date'] < current_date:
                        # Calculate PnL
                        exit_price = open_price * (1 - self.slippage_pct)
                        if pos['direction'] == SignalDirection.SHORT:
                            pnl = (pos['entry_price'] - exit_price) * pos['shares']
                        else:
                            pnl = (exit_price - pos['entry_price']) * pos['shares']

                        # Subtract commission
                        commission = exit_price * pos['shares'] * self.commission_pct
                        pnl -= commission

                        pnl_pct = pnl / (pos['entry_price'] * pos['shares'])

                        # Record trade
                        trade = OpExTradeRecord(
                            symbol=symbol,
                            entry_date=pos['entry_date'],
                            exit_date=current_date,
                            entry_price=pos['entry_price'],
                            exit_price=exit_price,
                            direction=pos['direction'],
                            phase=pos['phase'],
                            pin_strike=pos['pin_strike'],
                            distance_to_pin=pos['distance_to_pin'],
                            net_gex=pos['net_gex'],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            shares=pos['shares'],
                        )
                        trades.append(trade)

                        capital += pnl
                        del positions[symbol]

                # Check for entry
                if symbol not in positions and phase in (OpExPhase.PINNING, OpExPhase.POST_OPEX):
                    # Get GEX data
                    gex_summary = gex_lookup.get((symbol, current_date))

                    if gex_summary:
                        pin_strike = gex_summary.pin_strike
                        net_gex = gex_summary.total_gex
                    else:
                        # Estimate pin strike
                        pin_strike = round(close_price / 5) * 5
                        net_gex = 1_000_000  # Assume positive

                    # Check conditions - use absolute GEX (sign depends on dealer assumption)
                    # We want high GEX magnitude for pinning potential
                    if abs(net_gex) < 1_000_000:  # Minimum GEX threshold
                        continue

                    distance_to_pin = (close_price - pin_strike) / pin_strike if pin_strike else 0
                    abs_distance = abs(distance_to_pin)

                    min_dist = self.strategy.min_distance_pct
                    max_dist = self.strategy.max_distance_pct

                    if abs_distance < min_dist or abs_distance > max_dist:
                        continue

                    # Determine direction
                    if phase == OpExPhase.PINNING:
                        # Fade: go opposite to distance
                        direction = SignalDirection.SHORT if distance_to_pin > 0 else SignalDirection.LONG
                    else:  # POST_OPEX
                        # Momentum: go same direction
                        direction = SignalDirection.LONG if distance_to_pin > 0 else SignalDirection.SHORT

                    # Only take long positions for now (simpler)
                    if direction != SignalDirection.LONG:
                        continue

                    # Calculate position size
                    position_value = capital * self.position_size_pct
                    entry_price = close_price * (1 + self.slippage_pct)
                    shares = int(position_value / entry_price)

                    if shares <= 0:
                        continue

                    # Commission on entry
                    commission = entry_price * shares * self.commission_pct

                    # Open position
                    positions[symbol] = {
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'direction': direction,
                        'phase': phase,
                        'pin_strike': pin_strike,
                        'distance_to_pin': distance_to_pin,
                        'net_gex': net_gex,
                    }

                    capital -= commission

            # Record daily equity
            equity.append(capital)
            equity_dates.append(current_date)

        # Create results
        results = OpExBacktestResults(trades=trades)

        # Create equity curve
        if equity_dates:
            results.equity_curve = pd.Series(equity, index=pd.to_datetime(equity_dates))
            results.daily_returns = results.equity_curve.pct_change().dropna()

        return results

    def _calculate_metrics(self, results: OpExBacktestResults) -> None:
        """Calculate performance metrics."""
        trades = results.trades

        if not trades:
            return

        # Basic counts
        results.total_trades = len(trades)
        results.winning_trades = sum(1 for t in trades if t.pnl > 0)
        results.losing_trades = sum(1 for t in trades if t.pnl <= 0)
        results.win_rate = results.winning_trades / results.total_trades if results.total_trades else 0

        # PnL stats
        pnls = [t.pnl for t in trades]
        results.total_pnl = sum(pnls)
        results.avg_pnl = np.mean(pnls) if pnls else 0

        winners = [t.pnl for t in trades if t.pnl > 0]
        losers = [t.pnl for t in trades if t.pnl <= 0]
        results.avg_winner = np.mean(winners) if winners else 0
        results.avg_loser = np.mean(losers) if losers else 0

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        results.profit_factor = gross_profit / gross_loss if gross_loss else 0

        # Sharpe ratio (annualized)
        if results.daily_returns is not None and len(results.daily_returns) > 0:
            daily_std = results.daily_returns.std()
            if daily_std > 0:
                results.sharpe_ratio = (results.daily_returns.mean() / daily_std) * np.sqrt(252)

        # Drawdown
        if results.equity_curve is not None:
            peak = results.equity_curve.expanding().max()
            drawdown = results.equity_curve - peak
            results.max_drawdown = drawdown.min()
            results.max_drawdown_pct = (drawdown / peak).min()

        # Phase breakdown
        pinning_trades = [t for t in trades if t.phase == OpExPhase.PINNING]
        post_opex_trades = [t for t in trades if t.phase == OpExPhase.POST_OPEX]

        results.pinning_trades = len(pinning_trades)
        results.pinning_win_rate = (
            sum(1 for t in pinning_trades if t.pnl > 0) / len(pinning_trades)
            if pinning_trades else 0
        )

        results.post_opex_trades = len(post_opex_trades)
        results.post_opex_win_rate = (
            sum(1 for t in post_opex_trades if t.pnl > 0) / len(post_opex_trades)
            if post_opex_trades else 0
        )

    def run_walk_forward(
        self,
        start_date: str,
        end_date: str,
        train_months: int = 24,
        test_months: int = 6,
        gex_data: Optional[Dict[str, List[DailyGEXSummary]]] = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            start_date: Overall start date
            end_date: Overall end date
            train_months: Training period in months
            test_months: Test period in months
            gex_data: Pre-computed GEX data

        Returns:
            Dict with in-sample and out-of-sample results
        """
        from dateutil.relativedelta import relativedelta

        logger.info(f"\nRunning walk-forward validation")
        logger.info(f"  Train: {train_months} months, Test: {test_months} months")

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        in_sample_results = []
        out_of_sample_results = []

        current_start = start

        while current_start + relativedelta(months=train_months + test_months) <= end:
            train_end = current_start + relativedelta(months=train_months)
            test_end = train_end + relativedelta(months=test_months)

            # In-sample (training)
            train_result = self.run_backtest(
                start_date=current_start.strftime('%Y-%m-%d'),
                end_date=train_end.strftime('%Y-%m-%d'),
                gex_data=gex_data,
            )
            in_sample_results.append({
                'period': f"{current_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
                'sharpe': train_result.sharpe_ratio,
                'win_rate': train_result.win_rate,
                'total_pnl': train_result.total_pnl,
                'trades': train_result.total_trades,
            })

            # Out-of-sample (test)
            test_result = self.run_backtest(
                start_date=train_end.strftime('%Y-%m-%d'),
                end_date=test_end.strftime('%Y-%m-%d'),
                gex_data=gex_data,
            )
            out_of_sample_results.append({
                'period': f"{train_end.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
                'sharpe': test_result.sharpe_ratio,
                'win_rate': test_result.win_rate,
                'total_pnl': test_result.total_pnl,
                'trades': test_result.total_trades,
            })

            # Move to next period
            current_start = train_end

        # Calculate summary
        is_sharpes = [r['sharpe'] for r in in_sample_results if r['sharpe']]
        oos_sharpes = [r['sharpe'] for r in out_of_sample_results if r['sharpe']]

        is_avg_sharpe = np.mean(is_sharpes) if is_sharpes else 0
        oos_avg_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
        degradation = (is_avg_sharpe - oos_avg_sharpe) / is_avg_sharpe if is_avg_sharpe else 0

        return {
            'in_sample': in_sample_results,
            'out_of_sample': out_of_sample_results,
            'is_avg_sharpe': is_avg_sharpe,
            'oos_avg_sharpe': oos_avg_sharpe,
            'degradation': degradation,
        }

    def print_results(self, results: OpExBacktestResults) -> None:
        """Print formatted backtest results."""
        print("\n" + "=" * 60)
        print("OPEX PINNING STRATEGY BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nTrade Summary:")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Winning: {results.winning_trades} ({results.win_rate:.1%})")
        print(f"  Losing: {results.losing_trades}")

        print(f"\nPnL Summary:")
        print(f"  Total PnL: ${results.total_pnl:,.2f}")
        print(f"  Avg Trade: ${results.avg_pnl:,.2f}")
        print(f"  Avg Winner: ${results.avg_winner:,.2f}")
        print(f"  Avg Loser: ${results.avg_loser:,.2f}")
        print(f"  Profit Factor: {results.profit_factor:.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.1%})")

        print(f"\nPhase Breakdown:")
        print(f"  PINNING: {results.pinning_trades} trades, {results.pinning_win_rate:.1%} win rate")
        print(f"  POST_OPEX: {results.post_opex_trades} trades, {results.post_opex_win_rate:.1%} win rate")

        print("=" * 60)
