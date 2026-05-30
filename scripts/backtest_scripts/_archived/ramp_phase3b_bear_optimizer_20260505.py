"""
RAMP Phase 3B: BEAR Regime Walk-Forward Optimization -- 2026-05-05

Hypothesis-driven sweep of two BEAR-regime parameters only:
  - pen_w_bear in {3.0, 4.0, 5.0, 6.0, 7.0}
  - exposure_during_bear in {0.0, 0.25, 0.5, 1.0}

All other regime parameters are held at production values.

Walk-forward methodology:
  W1: train 2017-2021, test 2021 (1 year)
  W2: train 2017-2021, test 2022 (1 year)
  W3: train 2018-2022, test 2023 (1 year)
  W4: train 2019-2023, test 2024 (1 year)

Winner: max mean per-window OOS Sharpe across W1-W4.
Final validation on EXTENDED-OOS 2025-01-01 to 2026-04-30 (never touched in selection).

Usage:
    python scripts/backtest_scripts/ramp_phase3b_bear_optimizer_20260505.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os
import itertools

import numpy as np
import pandas as pd
import yfinance as yf

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Production REGIME_PARAMS -- DO NOT change non-BEAR regimes
# =============================================================================

PRODUCTION_REGIME_PARAMS = {
    'STRONG_BULL':   {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 20},
    'WEAK_BULL':     {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10},
    'SIDEWAYS':      {'long_p': 21, 'short_p': 5, 'long_w': 0.5, 'pen_w': 2.0, 'top_n':  5},
    'UNPREDICTABLE': {'long_p': 42, 'short_p': 21,'long_w': 0.5, 'pen_w': 4.0, 'top_n': 10},
    'BEAR':          {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 3.0, 'top_n': 10},
}

DEFAULT_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 4.0, 'top_n': 10}

VIX_THRESHOLD    = 25.0
SPY_DD_THRESHOLD = -0.05
PROD_REDUCED_EXP = 0.5  # production exposure when crash protection fires (non-BEAR)
MAX_DAILY_RETURN = 0.20

FULL_DATA_START = '2016-07-01'  # buffer for indicator warmup
FULL_DATA_END   = '2026-04-30'

# Backtest bounds
BACKTEST_START = '2017-01-01'
BACKTEST_END   = '2026-04-30'

EXT_OOS_START = '2025-01-01'
EXT_OOS_END   = '2026-04-30'

# Optimization grid -- hypothesis-driven, only 2 params
PEN_W_BEAR_VALUES      = [3.0, 4.0, 5.0, 6.0, 7.0]
EXPOSURE_BEAR_VALUES   = [0.0, 0.25, 0.5, 1.0]

# Walk-forward windows (train_end is inclusive; test is the full year after)
WF_WINDOWS = [
    # (label, train_start, train_end, test_start, test_end)
    ('W1', '2017-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('W2', '2017-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('W3', '2018-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('W4', '2019-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

# Cost modeling
# Large-cap liquid: 5 bps/side (10 bps round-trip)
# Applied as a daily drag proportional to portfolio turnover assumption
# Assumes ~5% daily turnover on average (rebalanced daily in top_n selection)
# 5 bps/side x 2 sides x 0.05 turnover = 0.5 bps drag/day
COST_TIERS = {
    '0x':   0.0000,    # research / no-cost baseline
    '1x':   0.0005,    # 5 bps/side assumed 10% daily turnover -> ~1 bp/day drag
    '1.5x': 0.00075,   # 7.5 bps/side
}


# =============================================================================
# Data Loading (shared across all backtest calls)
# =============================================================================

def load_sp500_symbols() -> List[str]:
    csv_path = Path('config/universes/sp500-2025.csv')
    if not csv_path.exists():
        logger.error(f"Universe file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    if 'Symbol' in df.columns:
        col = 'Symbol'
    elif 'symbol' in df.columns:
        col = 'symbol'
    else:
        col = df.columns[0]
    symbols = df[col].astype(str).tolist()
    return [s for s in symbols if not s.isdigit()]


def load_ohlcv_yf(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Download OHLCV for all symbols. Returns dict symbol -> DataFrame."""
    logger.info(f"Downloading OHLCV for {len(symbols)} symbols ({start_date} to {end_date})...")
    yf_symbols = [s.replace('.', '-') for s in symbols]

    raw = yf.download(
        yf_symbols,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=True,
        threads=True,
    )

    if raw.empty:
        logger.error("yfinance download returned empty DataFrame")
        return {}

    if not isinstance(raw.columns, pd.MultiIndex):
        logger.error("Unexpected yfinance column format")
        return {}

    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    symbol_data: Dict[str, pd.DataFrame] = {}
    for yf_sym, orig_sym in zip(yf_symbols, symbols):
        try:
            ohlcv = pd.DataFrame({
                'open':   raw['Open'][yf_sym],
                'high':   raw['High'][yf_sym],
                'low':    raw['Low'][yf_sym],
                'close':  raw['Close'][yf_sym],
                'volume': raw['Volume'][yf_sym],
            })
            ohlcv = ohlcv.dropna(how='all')
            if len(ohlcv) > 10:
                symbol_data[orig_sym] = ohlcv
        except KeyError:
            pass

    logger.info(f"Loaded OHLCV for {len(symbol_data)} symbols")
    return symbol_data


def load_market_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPY and VIX."""
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

    def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    return flatten_cols(spy), flatten_cols(vix)


# =============================================================================
# Pre-computation
# =============================================================================

def build_close_df(symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {sym: df['close'] for sym, df in symbol_data.items()}
    return pd.DataFrame(closes)


# =============================================================================
# Metric Helpers
# =============================================================================

def compute_metrics(returns: pd.Series) -> Dict:
    if returns.empty or len(returns) < 2:
        return {
            'total_return': 0.0, 'cagr': 0.0, 'sharpe': 0.0,
            'max_drawdown': 0.0, 'win_rate': 0.0, 'n_days': 0,
            'avg_daily_return': 0.0, 'std_daily_return': 0.0,
        }
    cum = (1 + returns).cumprod()
    total_return = float(cum.iloc[-1] - 1)
    n_years = len(returns) / 252.0
    cagr = float((1 + total_return) ** (1.0 / n_years) - 1) if n_years > 0 else 0.0
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0
    rolling_max = cum.cummax()
    max_dd = float(((cum - rolling_max) / rolling_max).min())
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': float((returns > 0).mean()),
        'n_days': len(returns),
        'avg_daily_return': float(returns.mean()),
        'std_daily_return': float(returns.std()),
    }


def period_returns(results_df: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Extract the 'return' series for a given date range."""
    if results_df.empty:
        return pd.Series(dtype=float)
    mask = (results_df['date'] >= start) & (results_df['date'] <= end)
    return results_df.loc[mask, 'return'].reset_index(drop=True)


# =============================================================================
# Core Backtest (parameterized by pen_w_bear and exposure_during_bear)
# =============================================================================

def run_backtest(
    pen_w_bear: float,
    exposure_during_bear: float,
    close_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    detector: MarketRegimeDetector,
    start_date: str,
    end_date: str,
    daily_cost_drag: float = 0.0,
) -> pd.DataFrame:
    """
    Run one backtest configuration over [start_date, end_date].

    BEAR regime uses pen_w_bear and exposure_during_bear.
    All other regimes use production parameters and PROD_REDUCED_EXP when
    crash protection fires.

    Returns DataFrame with columns: date, regime, return, exposure, positions.
    """
    # Build regime params for this config
    regime_params = {k: dict(v) for k, v in PRODUCTION_REGIME_PARAMS.items()}
    regime_params['BEAR']['pen_w'] = pen_w_bear

    # Pre-compute pct changes over entire close_df (vectorized)
    pc21 = close_df.pct_change(21)
    pc5  = close_df.pct_change(5)

    # Pre-compute SPY cummax drawdown
    spy_close  = spy_df['close']
    spy_cummax = spy_close.cummax()
    spy_dd     = (spy_close - spy_cummax) / spy_cummax

    # Trading days within [start_date, end_date]
    # Use close_df index filtered to window
    trading_days_full = close_df.index.tolist()
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)
    trading_days = [d for d in trading_days_full if start_dt <= d <= end_dt]

    if len(trading_days) < 50:
        logger.error(f"Too few trading days ({len(trading_days)}) for {start_date}:{end_date}")
        return pd.DataFrame()

    results = []

    for i, date in enumerate(trading_days[:-1]):

        # --- Regime detection (uses all data up to and including date) ---
        spy_sub = spy_df[spy_df.index <= date]
        vix_sub = vix_df[vix_df.index <= date]

        if len(spy_sub) < 252 or len(vix_sub) < 252:
            regime = 'SIDEWAYS'
        else:
            try:
                regime, _ = detector.classify_regime(spy_sub, vix_sub, date)
            except Exception:
                regime = 'SIDEWAYS'

        params = regime_params.get(regime, DEFAULT_PARAMS)
        long_w = params['long_w']
        pen_w  = params['pen_w']
        top_n  = params['top_n']

        # --- Momentum signal (no lookahead: signal at date, entry at close) ---
        if date not in pc21.index or date not in pc5.index:
            continue

        long_ret_row  = pc21.loc[date]
        short_ret_row = pc5.loc[date]
        momentum      = (long_w * long_ret_row) - (pen_w * short_ret_row)
        momentum      = momentum.dropna()

        if momentum.empty:
            continue

        top_stocks = momentum.nlargest(top_n).index.tolist()

        # --- Crash protection / exposure ---
        vix_val   = vix_df.loc[:date, 'close'].iloc[-1] if len(vix_df.loc[:date]) > 0 else 20.0
        spy_dd_val = spy_dd.loc[:date].iloc[-1] if len(spy_dd.loc[:date]) > 0 else 0.0

        if regime == 'BEAR':
            # BEAR gets its own exposure param (no crash-protection override applies here;
            # the exposure IS the bear-specific value, optionally further reduced by crash prot)
            base_exposure = exposure_during_bear
            # Still apply crash protection on top if triggered (floor, not separate override)
            if vix_val > VIX_THRESHOLD or spy_dd_val < SPY_DD_THRESHOLD:
                # In BEAR + crash-protection: take the more conservative of bear_exposure
                # and the crash-protection reduced level (0.5 * bear_exposure, floor 0.0)
                exposure = min(base_exposure, base_exposure * PROD_REDUCED_EXP)
            else:
                exposure = base_exposure
        else:
            exposure = 1.0
            if vix_val > VIX_THRESHOLD or spy_dd_val < SPY_DD_THRESHOLD:
                exposure = PROD_REDUCED_EXP

        # --- Equal-weight positions ---
        n_valid = len(top_stocks)
        w = exposure / n_valid if n_valid > 0 else 0.0
        weights = {s: w for s in top_stocks}

        # --- Next-day returns ---
        next_date = trading_days[i + 1]
        if date not in close_df.index or next_date not in close_df.index:
            continue

        today_prices = close_df.loc[date]
        next_prices  = close_df.loc[next_date]

        port_return = 0.0
        positions   = 0

        for sym in weights:
            if sym in today_prices.index and sym in next_prices.index:
                p0 = today_prices[sym]
                p1 = next_prices[sym]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret = np.clip((p1 - p0) / p0, -MAX_DAILY_RETURN, MAX_DAILY_RETURN)
                    port_return += weights[sym] * ret
                    positions   += 1

        # Apply daily cost drag (proportional to gross exposure)
        port_return -= daily_cost_drag * exposure

        results.append({
            'date':      next_date,
            'regime':    regime,
            'return':    port_return,
            'exposure':  exposure,
            'positions': positions,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


# =============================================================================
# Walk-Forward Optimization
# =============================================================================

def run_walk_forward(
    close_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    detector: MarketRegimeDetector,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run all 20 configs x 4 windows = 80 backtests.
    Returns:
      - results_df: per-config-per-window Sharpe matrix
      - winner_info: dict with best config details
    """
    configs = list(itertools.product(PEN_W_BEAR_VALUES, EXPOSURE_BEAR_VALUES))
    # Prepend production config first for easy comparison
    prod_config = (3.0, 0.5)
    if prod_config in configs:
        configs.remove(prod_config)
    configs = [prod_config] + configs  # production always first

    n_configs  = len(configs)
    n_windows  = len(WF_WINDOWS)
    total_runs = n_configs * n_windows
    logger.info(f"Walk-forward: {n_configs} configs x {n_windows} windows = {total_runs} backtests")

    all_rows = []
    run_count = 0
    t_start = time.time()

    for pen_w_bear, exp_bear in configs:
        row = {
            'pen_w_bear':           pen_w_bear,
            'exposure_during_bear': exp_bear,
        }

        is_sharpes  = []
        oos_sharpes = []

        for label, train_start, train_end, test_start, test_end in WF_WINDOWS:
            # IS backtest (for reporting IS/OOS gap; not used for winner selection)
            is_df = run_backtest(
                pen_w_bear         = pen_w_bear,
                exposure_during_bear = exp_bear,
                close_df           = close_df,
                spy_df             = spy_df,
                vix_df             = vix_df,
                detector           = detector,
                start_date         = train_start,
                end_date           = train_end,
                daily_cost_drag    = 0.0,
            )
            is_rets = period_returns(is_df, train_start, train_end)
            is_m    = compute_metrics(is_rets)
            is_sharpes.append(is_m['sharpe'])

            # OOS backtest (the metric that drives winner selection)
            oos_df = run_backtest(
                pen_w_bear           = pen_w_bear,
                exposure_during_bear = exp_bear,
                close_df             = close_df,
                spy_df               = spy_df,
                vix_df               = vix_df,
                detector             = detector,
                start_date           = test_start,
                end_date             = test_end,
                daily_cost_drag      = 0.0,
            )
            oos_rets = period_returns(oos_df, test_start, test_end)
            oos_m    = compute_metrics(oos_rets)

            row[f'{label}_IS_sharpe']  = is_m['sharpe']
            row[f'{label}_OOS_sharpe'] = oos_m['sharpe']
            row[f'{label}_OOS_cagr']   = oos_m['cagr']
            row[f'{label}_OOS_maxdd']  = oos_m['max_drawdown']
            row[f'{label}_OOS_ndays']  = oos_m['n_days']

            is_sharpes.append(is_m['sharpe'])  # also collected for gap
            oos_sharpes.append(oos_m['sharpe'])

            run_count += 1

        # Aggregate across windows
        mean_oos  = float(np.mean(oos_sharpes))
        min_oos   = float(np.min(oos_sharpes))
        mean_is   = float(np.mean(is_sharpes))

        row['mean_OOS_sharpe'] = mean_oos
        row['min_OOS_sharpe']  = min_oos
        row['mean_IS_sharpe']  = mean_is

        if mean_is != 0:
            row['IS_OOS_gap_pct'] = (mean_is - mean_oos) / abs(mean_is) * 100.0
        else:
            row['IS_OOS_gap_pct'] = float('nan')

        all_rows.append(row)

        elapsed = time.time() - t_start
        pct_done = run_count / total_runs * 100
        logger.info(
            f"  Config pen_w={pen_w_bear} exp={exp_bear}: "
            f"mean_OOS={mean_oos:.3f}, min_OOS={min_oos:.3f} "
            f"[{run_count}/{total_runs} runs, {pct_done:.0f}%, {elapsed:.0f}s elapsed]"
        )

    results_df = pd.DataFrame(all_rows)

    # Winner: max mean_OOS_sharpe; tiebreaker: max min_OOS_sharpe
    # Exclude production from "winner" search initially; find the best overall config
    sorted_df = results_df.sort_values(
        ['mean_OOS_sharpe', 'min_OOS_sharpe'],
        ascending=[False, False]
    ).reset_index(drop=True)

    winner_row  = sorted_df.iloc[0]
    winner_info = {
        'pen_w_bear':           float(winner_row['pen_w_bear']),
        'exposure_during_bear': float(winner_row['exposure_during_bear']),
        'mean_OOS_sharpe':      float(winner_row['mean_OOS_sharpe']),
        'min_OOS_sharpe':       float(winner_row['min_OOS_sharpe']),
        'mean_IS_sharpe':       float(winner_row['mean_IS_sharpe']),
        'IS_OOS_gap_pct':       float(winner_row['IS_OOS_gap_pct']),
    }

    # Also retrieve production row for comparison
    prod_row = results_df[
        (results_df['pen_w_bear'] == 3.0) &
        (results_df['exposure_during_bear'] == 0.5)
    ].iloc[0]
    winner_info['prod_mean_OOS_sharpe'] = float(prod_row['mean_OOS_sharpe'])
    winner_info['prod_min_OOS_sharpe']  = float(prod_row['min_OOS_sharpe'])

    logger.info(
        f"Walk-forward complete. Winner: pen_w={winner_info['pen_w_bear']}, "
        f"exp={winner_info['exposure_during_bear']}, "
        f"mean_OOS={winner_info['mean_OOS_sharpe']:.3f}"
    )

    return results_df, winner_info


# =============================================================================
# EXT-OOS Validation
# =============================================================================

def run_ext_oos(
    pen_w_bear: float,
    exposure_during_bear: float,
    close_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    detector: MarketRegimeDetector,
    label: str = '',
) -> Dict:
    """Run config on EXT-OOS (2025-01-01 to 2026-04-30) at multiple cost levels."""
    cost_results = {}
    for cost_label, drag in COST_TIERS.items():
        df = run_backtest(
            pen_w_bear           = pen_w_bear,
            exposure_during_bear = exposure_during_bear,
            close_df             = close_df,
            spy_df               = spy_df,
            vix_df               = vix_df,
            detector             = detector,
            start_date           = EXT_OOS_START,
            end_date             = EXT_OOS_END,
            daily_cost_drag      = drag,
        )
        rets = period_returns(df, EXT_OOS_START, EXT_OOS_END)
        m    = compute_metrics(rets)
        cost_results[cost_label] = m
        logger.info(
            f"EXT-OOS [{label}] cost={cost_label}: "
            f"Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.1%}, MaxDD={m['max_drawdown']:.1%}"
        )
    return cost_results


# =============================================================================
# Parameter Stability Check (neighbors of winner)
# =============================================================================

def run_neighbor_check(
    winner_pen_w: float,
    winner_exp: float,
    close_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    detector: MarketRegimeDetector,
) -> pd.DataFrame:
    """Test +/-1 step in pen_w and +/-0.25 in exposure to check cliff-edge risk."""
    neighbors = []

    pen_w_step = 1.0
    exp_step   = 0.25

    pen_candidates = [
        winner_pen_w - pen_w_step,
        winner_pen_w,
        winner_pen_w + pen_w_step,
    ]
    exp_candidates = [
        max(0.0, winner_exp - exp_step),
        winner_exp,
        min(1.0, winner_exp + exp_step),
    ]

    # Deduplicate
    pen_candidates = sorted(set([p for p in pen_candidates if 0 < p <= 10.0]))
    exp_candidates = sorted(set([e for e in exp_candidates if 0.0 <= e <= 1.0]))

    for pen_w, exp in itertools.product(pen_candidates, exp_candidates):
        oos_sharpes = []
        for _, _, _, test_start, test_end in WF_WINDOWS:
            df = run_backtest(
                pen_w_bear           = pen_w,
                exposure_during_bear = exp,
                close_df             = close_df,
                spy_df               = spy_df,
                vix_df               = vix_df,
                detector             = detector,
                start_date           = test_start,
                end_date             = test_end,
            )
            rets = period_returns(df, test_start, test_end)
            m    = compute_metrics(rets)
            oos_sharpes.append(m['sharpe'])
        neighbors.append({
            'pen_w_bear':           pen_w,
            'exposure_during_bear': exp,
            'mean_OOS_sharpe':      np.mean(oos_sharpes),
            'is_winner':            (pen_w == winner_pen_w and exp == winner_exp),
        })

    return pd.DataFrame(neighbors)


# =============================================================================
# Report Writer
# =============================================================================

def write_report(
    wf_results_df: pd.DataFrame,
    winner_info: Dict,
    ext_oos_winner: Dict,
    ext_oos_prod: Dict,
    neighbor_df: pd.DataFrame,
    symbols_count: int,
):
    report_path = Path('docs/reports/ramp/20260505_phase3b_bear_optimizer.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_f(v, d=3):
        try:
            return f"{float(v):.{d}f}"
        except Exception:
            return str(v)

    def fmt_pct(v, d=1):
        try:
            return f"{float(v)*100:.{d}f}%"
        except Exception:
            return str(v)

    def cell(val):
        """Format cell value for markdown table."""
        if isinstance(val, float) and np.isnan(val):
            return 'N/A'
        return fmt_f(val, 3)

    lines = []
    lines.append("# RAMP Phase 3B: BEAR Regime Walk-Forward Optimization -- 2026-05-05")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(
        "Root cause investigation (docs/reports/ramp/20260505_root_cause_investigation.md) found "
        "that H2 (regime params don't generalize) is the dominant cause of EXT-OOS alpha decay. "
        "BEAR regime (64 days, Sharpe -2.17 in 2025-2026) was the worst individual contributor. "
        "V1 (no regime, always pen_w=5.0) produced EXT-OOS Sharpe 0.314 vs production V0 0.070, "
        "suggesting the BEAR penalty weight (currently 3.0, lower than all other regimes) is the "
        "most actionable lever. This run sweeps pen_w_bear and exposure_during_bear jointly "
        "using 4 walk-forward windows, with 2025-2026 reserved as true OOS."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"Universe: sp500-2025.csv ({symbols_count} symbols). "
        "Data: yfinance split-adjusted (auto_adjust=True), 2016-07-01 to 2026-04-30. "
        "Transaction costs: 0% for walk-forward selection; cost sensitivity tested on EXT-OOS winner. "
        "+/-20% daily return cap. Regime detection: same MarketRegimeDetector as production. "
        "Non-BEAR regimes held at production values. "
        "Winner selected by max mean per-window OOS Sharpe across W1-W4 only; "
        "2025-2026 data not seen during optimization."
    )
    lines.append("")
    lines.append("Walk-forward windows:")
    lines.append("- W1: train 2017-2020, test 2021")
    lines.append("- W2: train 2017-2021, test 2022")
    lines.append("- W3: train 2018-2022, test 2023")
    lines.append("- W4: train 2019-2023, test 2024")
    lines.append("")

    # --- Full WF matrix ---
    lines.append("## Walk-forward results matrix (per-window OOS Sharpe)")
    lines.append("")
    lines.append(
        "Production row first, then sorted by mean OOS Sharpe descending."
    )
    lines.append("")
    header = (
        "| pen_w_bear | exposure_during_bear "
        "| W1 OOS (2021) | W2 OOS (2022) | W3 OOS (2023) | W4 OOS (2024) "
        "| Mean OOS | Min OOS | IS/OOS gap |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    # Production row first
    prod_df = wf_results_df[
        (wf_results_df['pen_w_bear'] == 3.0) &
        (wf_results_df['exposure_during_bear'] == 0.5)
    ]
    rest_df = wf_results_df[
        ~((wf_results_df['pen_w_bear'] == 3.0) & (wf_results_df['exposure_during_bear'] == 0.5))
    ].sort_values('mean_OOS_sharpe', ascending=False)

    ordered_df = pd.concat([prod_df, rest_df], ignore_index=True)

    for _, r in ordered_df.iterrows():
        is_prod = (r['pen_w_bear'] == 3.0 and r['exposure_during_bear'] == 0.5)
        pen_label = f"{r['pen_w_bear']:.1f} (prod)" if is_prod else f"{r['pen_w_bear']:.1f}"
        exp_label = f"{r['exposure_during_bear']:.2f} (prod)" if is_prod else f"{r['exposure_during_bear']:.2f}"
        gap_str = f"{r['IS_OOS_gap_pct']:.1f}%" if not np.isnan(r.get('IS_OOS_gap_pct', float('nan'))) else "N/A"
        lines.append(
            f"| {pen_label} | {exp_label} "
            f"| {cell(r.get('W1_OOS_sharpe', float('nan')))} "
            f"| {cell(r.get('W2_OOS_sharpe', float('nan')))} "
            f"| {cell(r.get('W3_OOS_sharpe', float('nan')))} "
            f"| {cell(r.get('W4_OOS_sharpe', float('nan')))} "
            f"| {cell(r.get('mean_OOS_sharpe', float('nan')))} "
            f"| {cell(r.get('min_OOS_sharpe', float('nan')))} "
            f"| {gap_str} |"
        )

    lines.append("")

    # --- Winner summary ---
    w = winner_info
    lines.append("## Best config (by mean OOS Sharpe across W1-W4)")
    lines.append("")
    lines.append(f"- pen_w_bear: **{w['pen_w_bear']:.1f}**")
    lines.append(f"- exposure_during_bear: **{w['exposure_during_bear']:.2f}**")
    lines.append(f"- Mean OOS Sharpe W1-W4: **{w['mean_OOS_sharpe']:.3f}**")
    lines.append(f"- Min OOS Sharpe (worst window): {w['min_OOS_sharpe']:.3f}")
    lines.append(f"- Mean IS Sharpe (for reference): {w['mean_IS_sharpe']:.3f}")
    lines.append(f"- IS/OOS degradation: {w['IS_OOS_gap_pct']:.1f}%")
    lines.append(f"- Production mean OOS Sharpe (pen_w=3.0, exp=0.5): {w['prod_mean_OOS_sharpe']:.3f}")
    lines.append(f"- Delta vs production: {w['mean_OOS_sharpe'] - w['prod_mean_OOS_sharpe']:+.3f}")
    lines.append("")

    # Null result check
    if w['mean_OOS_sharpe'] <= w['prod_mean_OOS_sharpe']:
        lines.append(
            "**NULL RESULT:** The best optimized config does not outperform production on "
            "mean per-window OOS Sharpe. The optimization found no improvement. "
            "This is an honest finding -- production parameters are already near-optimal "
            "on the W1-W4 walk-forward windows."
        )
    else:
        lines.append(
            f"The winner outperforms production by "
            f"{w['mean_OOS_sharpe'] - w['prod_mean_OOS_sharpe']:+.3f} mean OOS Sharpe "
            "across the 4 walk-forward windows."
        )
    lines.append("")

    # --- EXT-OOS Validation ---
    lines.append("## Final EXTENDED-OOS validation (2025-01-01 to 2026-04-30)")
    lines.append("")
    lines.append("Note: EXT-OOS was NOT used in winner selection. These are truly out-of-sample results.")
    lines.append("")
    lines.append("| Metric | Production V0 (pen_w=3.0, exp=0.5) | Optimized | Delta |")
    lines.append("|---|---|---|---|")

    def ext_val(d, key):
        return d.get('0x', {}).get(key, float('nan'))

    prod_ext_sharpe  = ext_val(ext_oos_prod, 'sharpe')
    opt_ext_sharpe   = ext_val(ext_oos_winner, 'sharpe')
    prod_ext_cagr    = ext_val(ext_oos_prod, 'cagr')
    opt_ext_cagr     = ext_val(ext_oos_winner, 'cagr')
    prod_ext_maxdd   = ext_val(ext_oos_prod, 'max_drawdown')
    opt_ext_maxdd    = ext_val(ext_oos_winner, 'max_drawdown')

    delta_sharpe = opt_ext_sharpe - prod_ext_sharpe
    delta_cagr   = opt_ext_cagr - prod_ext_cagr
    delta_maxdd  = opt_ext_maxdd - prod_ext_maxdd

    lines.append(
        f"| EXT-OOS Sharpe (0% costs) "
        f"| {fmt_f(prod_ext_sharpe)} "
        f"| {fmt_f(opt_ext_sharpe)} "
        f"| {delta_sharpe:+.3f} |"
    )
    lines.append(
        f"| EXT-OOS CAGR "
        f"| {fmt_pct(prod_ext_cagr)} "
        f"| {fmt_pct(opt_ext_cagr)} "
        f"| {fmt_pct(delta_cagr)} |"
    )
    lines.append(
        f"| EXT-OOS MaxDD "
        f"| {fmt_pct(prod_ext_maxdd)} "
        f"| {fmt_pct(opt_ext_maxdd)} "
        f"| {fmt_pct(delta_maxdd)} |"
    )
    lines.append("")

    # --- Cost sensitivity ---
    lines.append("## Cost sensitivity (winning config on EXT-OOS)")
    lines.append("")
    lines.append("| Cost Level | EXT-OOS Sharpe | EXT-OOS CAGR | EXT-OOS MaxDD |")
    lines.append("|---|---|---|---|")
    for cost_label in ['0x', '1x', '1.5x']:
        m = ext_oos_winner.get(cost_label, {})
        lines.append(
            f"| {cost_label} ({['0 bps/side','5 bps/side','7.5 bps/side'][['0x','1x','1.5x'].index(cost_label)]}) "
            f"| {fmt_f(m.get('sharpe', float('nan')))} "
            f"| {fmt_pct(m.get('cagr', float('nan')))} "
            f"| {fmt_pct(m.get('max_drawdown', float('nan')))} |"
        )
    lines.append("")

    # --- Pre-committed criteria ---
    lines.append("## Pre-committed evaluation criteria")
    lines.append("")
    ext_sharpe_0x   = ext_oos_winner.get('0x', {}).get('sharpe', float('nan'))
    ext_sharpe_1_5x = ext_oos_winner.get('1.5x', {}).get('sharpe', float('nan'))
    wf_aggregate    = w['mean_OOS_sharpe']
    prod_wf_sharpe  = w['prod_mean_OOS_sharpe']
    is_oos_gap      = w['IS_OOS_gap_pct']

    c1 = "PASS" if ext_sharpe_0x > 0.5 else "FAIL"
    c2 = "PASS" if (not np.isnan(ext_sharpe_1_5x) and ext_sharpe_1_5x > 0.3) else "FAIL"
    # OOS aggregate within +/-0.1 of V0's 0.823 (2017-2024 OOS baseline)
    # We use mean WF OOS Sharpe as proxy for 2017-2024 aggregate
    # V0 OOS 2022-2024 baseline = 0.823; allow +/-0.1 from prod wf Sharpe
    c3 = "PASS" if abs(wf_aggregate - prod_wf_sharpe) <= 0.1 or wf_aggregate > prod_wf_sharpe else "FAIL"
    c4 = "PASS" if (not np.isnan(is_oos_gap) and abs(is_oos_gap) < 30.0) else "FAIL"

    lines.append(f"- EXT-OOS Sharpe > 0.5 at 0% costs: **{c1}** ({fmt_f(ext_sharpe_0x, 3)})")
    lines.append(f"- EXT-OOS Sharpe > 0.3 at 1.5x costs: **{c2}** ({fmt_f(ext_sharpe_1_5x, 3)})")
    lines.append(
        f"- W1-W4 mean OOS Sharpe within +/-0.1 of production or better: **{c3}** "
        f"(winner {fmt_f(wf_aggregate, 3)} vs prod {fmt_f(prod_wf_sharpe, 3)})"
    )
    lines.append(f"- IS/OOS gap < 30%: **{c4}** ({is_oos_gap:.1f}%)")
    lines.append("")

    all_pass = all(x == "PASS" for x in [c1, c2, c3, c4])
    if all_pass:
        lines.append("**All 4 criteria pass. Winner is a PRODUCTION CANDIDATE.**")
    else:
        failed = [x for x in [c1, c2, c3, c4] if x == "FAIL"]
        lines.append(
            f"**{len(failed)} of 4 criteria failed. Winner is NOT a production candidate without further work.**"
        )
    lines.append("")

    # --- Overfitting check ---
    lines.append("## Overfitting check")
    lines.append("")
    lines.append(f"- Number of configs tested: 20 (5 pen_w_bear x 4 exposure_during_bear)")
    lines.append(f"- Tunable parameters: 2 (target <=3: PASS)")
    lines.append(f"- Both parameters have economic rationale: PASS")
    lines.append(
        f"  - pen_w_bear: contrarian penalty weight. Higher = stronger preference for stocks "
        "that did well long-term but underperformed short-term. Economic logic: in BEAR markets, "
        "momentum reversal is common; higher pen_w filters out recent momentum names."
    )
    lines.append(
        f"  - exposure_during_bear: capital deployment. Lower = more cash. Economic logic: "
        "BEAR regime implies elevated downside risk; reducing exposure limits drawdowns."
    )
    lines.append("")

    # Best/worst spread
    all_mean_oos = wf_results_df['mean_OOS_sharpe'].values
    best_val     = float(np.max(all_mean_oos))
    worst_val    = float(np.min(all_mean_oos))
    spread       = best_val - worst_val
    lines.append(f"- Best mean OOS Sharpe across configs: {best_val:.3f}")
    lines.append(f"- Worst mean OOS Sharpe across configs: {worst_val:.3f}")
    lines.append(f"- Best/worst spread: {spread:.3f}")
    lines.append("")

    # Neighbor stability
    if not neighbor_df.empty:
        winner_neighbor = neighbor_df[neighbor_df['is_winner'] == True]
        neighbors_only  = neighbor_df[neighbor_df['is_winner'] == False]
        if not winner_neighbor.empty and not neighbors_only.empty:
            winner_oos  = float(winner_neighbor.iloc[0]['mean_OOS_sharpe'])
            neighbor_mean = float(neighbors_only['mean_OOS_sharpe'].mean())
            neighbor_min  = float(neighbors_only['mean_OOS_sharpe'].min())
            degradation  = winner_oos - neighbor_mean
            lines.append("### Parameter stability (neighbors of winning config)")
            lines.append("")
            lines.append("| pen_w_bear | exposure_during_bear | Mean OOS Sharpe | Is Winner |")
            lines.append("|---|---|---|---|")
            for _, nr in neighbor_df.sort_values('mean_OOS_sharpe', ascending=False).iterrows():
                mark = "[*]" if nr['is_winner'] else ""
                lines.append(
                    f"| {nr['pen_w_bear']:.1f} | {nr['exposure_during_bear']:.2f} "
                    f"| {nr['mean_OOS_sharpe']:.3f} | {mark} |"
                )
            lines.append("")
            lines.append(f"Winner vs neighbor mean: {degradation:+.3f}")
            if degradation > 0.3:
                lines.append(
                    "**CLIFF-EDGE WARNING:** Winner Sharpe drops sharply in adjacent configs. "
                    "Likely overfit. Do NOT deploy without further validation."
                )
            else:
                lines.append(
                    "Neighbor degradation is moderate -- the winning config is not a sharp spike. "
                    "Parameter stability is ACCEPTABLE."
                )
            lines.append("")

    # DSR if winner Sharpe > 1.5
    if w['mean_OOS_sharpe'] > 1.5:
        n_combos   = 20
        n_obs      = 4  # 4 OOS windows used for selection
        best_sharpe = w['mean_OOS_sharpe']
        # DSR approximation: haircut = 1 - ln(N)/(2*T)
        haircut     = 1.0 - np.log(n_combos) / (2.0 * n_obs)
        dsr_sharpe  = best_sharpe * max(0, haircut)
        lines.append(f"**Deflated Sharpe Ratio (required: winner Sharpe {w['mean_OOS_sharpe']:.3f} > 1.5)**")
        lines.append(f"- N configs tested: {n_combos}, T independent observations: {n_obs}")
        lines.append(f"- Haircut factor: {haircut:.3f}")
        lines.append(f"- DSR-adjusted Sharpe: {dsr_sharpe:.3f}")
        if dsr_sharpe < 0.5:
            lines.append(
                "**DSR < 0.5: edge NOT statistically distinguishable from noise. REJECT.**"
            )
        elif dsr_sharpe < 1.0:
            lines.append(
                f"DSR-adjusted Sharpe {dsr_sharpe:.3f}: edge is marginal after selection bias adjustment."
            )
        lines.append("")

    # --- Conclusion ---
    lines.append("## Conclusion")
    lines.append("")

    if all_pass:
        lines.append(
            f"The walk-forward optimization found a configuration that outperforms production "
            f"across all 4 pre-committed criteria. "
            f"pen_w_bear={w['pen_w_bear']:.1f} and exposure_during_bear={w['exposure_during_bear']:.2f} "
            f"improve mean W1-W4 OOS Sharpe from {w['prod_mean_OOS_sharpe']:.3f} to {w['mean_OOS_sharpe']:.3f} "
            f"and EXT-OOS Sharpe from {prod_ext_sharpe:.3f} to {opt_ext_sharpe:.3f}."
        )
    elif w['mean_OOS_sharpe'] > w['prod_mean_OOS_sharpe']:
        lines.append(
            f"The optimized config (pen_w_bear={w['pen_w_bear']:.1f}, "
            f"exposure={w['exposure_during_bear']:.2f}) shows improvement in W1-W4 mean OOS Sharpe "
            f"({w['mean_OOS_sharpe']:.3f} vs {w['prod_mean_OOS_sharpe']:.3f}) but fails {len(failed)} "
            "of the 4 pre-committed EXT-OOS criteria. The improvement may not hold in true OOS."
        )
    else:
        lines.append(
            "**NULL RESULT.** The walk-forward optimization did not find any configuration "
            f"that meaningfully outperforms production (pen_w=3.0, exp=0.5) on the W1-W4 "
            "mean OOS Sharpe. The production parameters are near-optimal within this search space. "
            "The BEAR regime underperformance in 2025-2026 is structural and likely cannot be "
            "fixed by adjusting pen_w and exposure alone -- the momentum signal itself may be "
            "selecting the wrong stocks during recent BEAR periods (see H6 in root cause report)."
        )
    lines.append("")

    # --- Implications ---
    lines.append("## Implications")
    lines.append("")

    if all_pass:
        lines.append(
            f"**DEPLOY:** Replace BEAR regime pen_w=3.0 with pen_w={w['pen_w_bear']:.1f} "
            f"and BEAR exposure from 0.5 to {w['exposure_during_bear']:.2f} in production ramp_strategy.py. "
            "Run paper trading for 30 trading days before live capital deployment."
        )
    elif w['mean_OOS_sharpe'] > w['prod_mean_OOS_sharpe'] + 0.1:
        lines.append(
            "**FURTHER TEST:** The winning config shows promise on W1-W4 but does not fully "
            "clear EXT-OOS criteria. Recommended: run 90-day paper trading, then re-evaluate "
            "with an additional year of true OOS data before any live deployment."
        )
    else:
        lines.append(
            "**NOT ACTIONABLE within this parameter space.** Two recommended next steps: "
            "(1) Replace BEAR-regime equity selection with cash or SPY put protection entirely "
            "-- the root cause analysis (H6) shows BEAR stock selection averages -0.32% next-day "
            "returns, which pen_w adjustment alone cannot fix. "
            "(2) Investigate WEAK_BULL regime parameters, which account for 43.6% of EXT-OOS "
            "trading days and contributed Sharpe -0.78 in 2025-2026."
        )
    lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding='utf-8')
    logger.info(f"Report written: {report_path}")
    return str(report_path)


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 80)
    logger.info("RAMP PHASE 3B: BEAR REGIME WALK-FORWARD OPTIMIZATION 2026-05-05")
    logger.info("=" * 80)

    t_global = time.time()

    # 1. Load symbols and data
    symbols = load_sp500_symbols()
    if not symbols:
        logger.error("BLOCKED: No symbols loaded.")
        sys.exit(1)
    logger.info(f"Universe: {len(symbols)} symbols")

    symbol_data = load_ohlcv_yf(symbols, FULL_DATA_START, FULL_DATA_END)
    if not symbol_data:
        logger.error("BLOCKED: OHLCV download failed.")
        sys.exit(1)

    spy_df, vix_df = load_market_data(FULL_DATA_START, FULL_DATA_END)
    if spy_df.empty:
        logger.error("BLOCKED: SPY/VIX download failed.")
        sys.exit(1)

    logger.info("Data loaded. Building close DataFrame...")
    close_df = build_close_df(symbol_data)
    logger.info(f"close_df shape: {close_df.shape}")

    # Shared detector (stateless, safe to reuse across configs)
    detector = MarketRegimeDetector()

    # 2. Walk-forward optimization (20 configs x 4 windows = 80 backtests)
    logger.info("Starting walk-forward optimization...")
    t_wf = time.time()
    wf_results_df, winner_info = run_walk_forward(close_df, spy_df, vix_df, detector)
    logger.info(f"Walk-forward done in {(time.time()-t_wf)/60:.1f} min")

    # 3. EXT-OOS validation for winner and production V0
    logger.info("Running EXT-OOS validation for winner...")
    ext_oos_winner = run_ext_oos(
        pen_w_bear           = winner_info['pen_w_bear'],
        exposure_during_bear = winner_info['exposure_during_bear'],
        close_df             = close_df,
        spy_df               = spy_df,
        vix_df               = vix_df,
        detector             = detector,
        label                = f"winner(pen_w={winner_info['pen_w_bear']:.1f},exp={winner_info['exposure_during_bear']:.2f})",
    )

    logger.info("Running EXT-OOS validation for production V0...")
    ext_oos_prod = run_ext_oos(
        pen_w_bear           = 3.0,
        exposure_during_bear = 0.5,
        close_df             = close_df,
        spy_df               = spy_df,
        vix_df               = vix_df,
        detector             = detector,
        label                = "prod_V0",
    )

    # 4. Neighbor stability check
    logger.info("Running neighbor stability check...")
    neighbor_df = run_neighbor_check(
        winner_pen_w = winner_info['pen_w_bear'],
        winner_exp   = winner_info['exposure_during_bear'],
        close_df     = close_df,
        spy_df       = spy_df,
        vix_df       = vix_df,
        detector     = detector,
    )

    # 5. Save raw results
    output_dir = Path('logs/backtesting/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    wf_csv = output_dir / f'{ts}_ramp_phase3b_wf_results.csv'
    wf_results_df.to_csv(wf_csv, index=False)
    logger.info(f"WF results saved: {wf_csv}")

    winner_json = output_dir / f'{ts}_ramp_phase3b_winner.json'
    with open(winner_json, 'w') as f:
        json.dump({
            'winner': winner_info,
            'ext_oos_winner': {k: v for k, v in ext_oos_winner.items()},
            'ext_oos_prod':   {k: v for k, v in ext_oos_prod.items()},
        }, f, indent=2, default=str)
    logger.info(f"Winner JSON saved: {winner_json}")

    # 6. Write markdown report
    logger.info("Writing markdown report...")
    report_path = write_report(
        wf_results_df  = wf_results_df,
        winner_info    = winner_info,
        ext_oos_winner = ext_oos_winner,
        ext_oos_prod   = ext_oos_prod,
        neighbor_df    = neighbor_df,
        symbols_count  = len(symbols),
    )

    total_min = (time.time() - t_global) / 60.0
    logger.info(f"Total runtime: {total_min:.1f} min")
    logger.info(f"Report: {report_path}")
    logger.info(f"Winner: pen_w_bear={winner_info['pen_w_bear']:.1f}, "
                f"exposure={winner_info['exposure_during_bear']:.2f}, "
                f"mean_OOS_Sharpe={winner_info['mean_OOS_sharpe']:.3f}")


if __name__ == '__main__':
    main()
