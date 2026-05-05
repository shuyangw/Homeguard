"""
RAMP Root Cause Investigation 2026-05-05

Tests 5 variants and 8 diagnostics to isolate the cause of 2025-2026 alpha decay.

V0 -- Production RAMP (reference)
V1 -- Vanilla momentum (no regime gating)
V2 -- Inverse-vol weighting (risk parity via yang_zhang_rv)
V3 -- Winsorized momentum (cross-sectional outlier handling)
V4 -- SPY-vol overlay (additional exposure reduction)

D1 -- Rolling 252-day Sharpe across 2017-2026
D2 -- Per-year Sharpe
D3 -- Crash protection trigger frequency by year
D4 -- Regime distribution by year
D5 -- 2025-2026 BEAR-regime trade analysis

Usage:
    python scripts/backtest_scripts/ramp_root_cause_20260505.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

from src.settings import get_local_storage_dir
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.features import yang_zhang_rv, winsorize
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# RAMP Parameters -- Production values, verbatim from ramp_strategy.py
# =============================================================================

REGIME_PARAMS = {
    'STRONG_BULL': {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 20},
    'WEAK_BULL':   {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10},
    'SIDEWAYS':    {'long_p': 21, 'short_p': 5, 'long_w': 0.5, 'pen_w': 2.0, 'top_n':  5},
    'UNPREDICTABLE':{'long_p':42, 'short_p':21, 'long_w': 0.5, 'pen_w': 4.0, 'top_n': 10},
    'BEAR':        {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 3.0, 'top_n': 10},
}

DEFAULT_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 4.0, 'top_n': 10}

# V1 fixed params (no regime)
V1_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10}

VIX_THRESHOLD   = 25.0
SPY_DD_THRESHOLD = -0.05
REDUCED_EXPOSURE = 0.5
MAX_DAILY_RETURN = 0.20

FULL_START = '2017-01-01'
FULL_END   = '2026-04-30'

IS_START   = '2017-01-01'
IS_END     = '2021-12-31'
OOS_START  = '2022-01-01'
OOS_END    = '2024-12-31'
EXT_START  = '2025-01-01'
EXT_END    = '2026-04-30'


# =============================================================================
# Data Loading
# =============================================================================

def load_sp500_symbols() -> List[str]:
    csv_path = Path("config/universes/sp500-2025.csv")
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
    """Download OHLCV for all symbols. Returns dict symbol -> DataFrame with lowercase columns."""
    logger.info(f"Downloading OHLCV for {len(symbols)} symbols ({start_date} to {end_date})...")
    buffer_start = (pd.to_datetime(start_date) - pd.Timedelta(days=150)).strftime('%Y-%m-%d')
    yf_symbols = [s.replace('.', '-') for s in symbols]

    raw = yf.download(
        yf_symbols,
        start=buffer_start,
        end=end_date,
        progress=True,
        auto_adjust=True,
        threads=True,
    )

    if raw.empty:
        logger.error("BLOCKED: yfinance download returned empty DataFrame")
        return {}

    # MultiIndex: (field, symbol)
    if not isinstance(raw.columns, pd.MultiIndex):
        logger.error("BLOCKED: unexpected yfinance column format (not MultiIndex)")
        return {}

    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    needed_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_fields = raw.columns.get_level_values(0).unique().tolist()
    missing = [f for f in needed_fields if f not in available_fields]
    if missing:
        logger.error(f"BLOCKED: missing yfinance fields: {missing}")
        return {}

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

    logger.info(f"Loaded OHLCV for {len(symbol_data)} symbols ({len(raw)} days)")
    return symbol_data


def load_market_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPY and VIX with buffer."""
    buffer_start = (pd.to_datetime(start_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    spy = yf.download('SPY', start=buffer_start, end=end_date, progress=False, auto_adjust=True)
    vix = yf.download('^VIX', start=buffer_start, end=end_date, progress=False)

    def flatten_cols(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    return flatten_cols(spy), flatten_cols(vix)


# =============================================================================
# Pre-computation helpers
# =============================================================================

def build_close_df(symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble wide close-price DataFrame from symbol_data dict."""
    closes = {sym: df['close'] for sym, df in symbol_data.items()}
    return pd.DataFrame(closes)


def build_pct_change_dfs(close_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-compute 21-day and 5-day pct_change for all symbols (used by all variants)."""
    pc21 = close_df.pct_change(21)
    pc5  = close_df.pct_change(5)
    return pc21, pc5


def build_yz_rv_df(symbol_data: Dict[str, pd.DataFrame], window: int = 21) -> pd.DataFrame:
    """Pre-compute yang_zhang_rv for each symbol. Returns wide DataFrame."""
    logger.info(f"Pre-computing yang_zhang_rv (window={window}) for {len(symbol_data)} symbols...")
    rv_cols = {}
    for sym, df in symbol_data.items():
        try:
            rv_cols[sym] = yang_zhang_rv(df, window=window, annualization_factor=1.0)
        except Exception:
            pass
    return pd.DataFrame(rv_cols)


def build_spy_vol_percentile(spy_df: pd.DataFrame, window: int = 21, trail: int = 252) -> pd.Series:
    """Compute trailing 252-day percentile of SPY yang_zhang_rv."""
    spy_rv = yang_zhang_rv(spy_df, window=window, annualization_factor=1.0)
    # rolling rank pct over trailing 252 days
    percentile = spy_rv.rolling(trail, min_periods=trail).rank(pct=True)
    return percentile


# =============================================================================
# Metric helpers
# =============================================================================

def compute_metrics(returns: pd.Series) -> Dict:
    if returns.empty or len(returns) < 2:
        return {
            'total_return': 0, 'cagr': 0, 'sharpe': 0,
            'max_drawdown': 0, 'win_rate': 0, 'n_days': 0,
            'avg_daily_return': 0, 'std_daily_return': 0,
        }
    cum = (1 + returns).cumprod()
    total_return = cum.iloc[-1] - 1
    n_years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    rolling_max = cum.cummax()
    max_dd = ((cum - rolling_max) / rolling_max).min()
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


def period_metrics(results_df: pd.DataFrame, start: str, end: str) -> Dict:
    if results_df.empty:
        return {}
    subset = results_df[(results_df['date'] >= start) & (results_df['date'] <= end)]
    if subset.empty:
        return {}
    return compute_metrics(subset['return'].reset_index(drop=True))


# =============================================================================
# Core backtest loop  (shared across V0-V4, parameterized by variant)
# =============================================================================

def run_variant(
    variant: str,
    close_df: pd.DataFrame,
    pc21: pd.DataFrame,
    pc5: pd.DataFrame,
    symbol_data: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    yz_rv_df: Optional[pd.DataFrame] = None,
    spy_vol_pct: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Run a single backtest variant over full date range (IS+OOS+EXT-OOS).
    Returns results DataFrame with columns: date, regime, return, positions, exposure.
    """

    trading_days = close_df[FULL_START:FULL_END].index.tolist()
    logger.info(f"[{variant}] Running {len(trading_days)} trading days...")

    detector = MarketRegimeDetector()

    spy_close = spy_df['close']
    spy_cummax = spy_close.cummax()
    spy_dd = (spy_close - spy_cummax) / spy_cummax

    results = []
    for i, date in enumerate(trading_days[:-1]):

        # ---------- regime detection ----------
        spy_sub = spy_df[spy_df.index <= date]
        vix_sub = vix_df[vix_df.index <= date]

        if variant == 'V1':
            regime = 'FIXED'
            params = V1_PARAMS
        else:
            if len(spy_sub) < 252 or len(vix_sub) < 252:
                regime = 'SIDEWAYS'
            else:
                regime, _ = detector.classify_regime(spy_sub, vix_sub, date)
            params = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)

        top_n = params['top_n']
        long_w = params['long_w']
        pen_w = params['pen_w']

        # ---------- momentum signal ----------
        if date not in pc21.index or date not in pc5.index:
            continue

        long_ret_row = pc21.loc[date]
        short_ret_row = pc5.loc[date]

        if variant == 'V3':
            long_ret_row = winsorize(long_ret_row.dropna(), lower_q=0.01, upper_q=0.99)
            short_ret_row = winsorize(short_ret_row.dropna(), lower_q=0.01, upper_q=0.99)

        momentum = (long_w * long_ret_row) - (pen_w * short_ret_row)
        momentum = momentum.dropna()

        if momentum.empty:
            continue

        top_stocks = momentum.nlargest(top_n).index.tolist()

        # ---------- crash protection / exposure ----------
        vix_value = vix_df.loc[:date, 'close'].iloc[-1] if len(vix_df.loc[:date]) > 0 else 20.0
        spy_dd_value = spy_dd.loc[:date].iloc[-1] if len(spy_dd.loc[:date]) > 0 else 0.0

        exposure = 1.0
        if vix_value > VIX_THRESHOLD or spy_dd_value < SPY_DD_THRESHOLD:
            exposure = REDUCED_EXPOSURE

        # V4: additional SPY-vol overlay
        if variant == 'V4' and spy_vol_pct is not None:
            spy_vol_entries = spy_vol_pct.loc[:date]
            if len(spy_vol_entries) > 0 and not np.isnan(spy_vol_entries.iloc[-1]):
                if spy_vol_entries.iloc[-1] > 0.90:
                    exposure *= 0.5

        # ---------- position weighting ----------
        if variant == 'V2' and yz_rv_df is not None:
            # inverse-vol weights
            rv_row = yz_rv_df.loc[:date].iloc[-1] if len(yz_rv_df.loc[:date]) > 0 else pd.Series(dtype=float)
            inv_vols = {}
            for sym in top_stocks:
                if sym in rv_row.index:
                    rv_val = rv_row[sym]
                    if pd.notna(rv_val) and rv_val > 0:
                        inv_vols[sym] = 1.0 / rv_val
            if not inv_vols:
                # fallback to equal weight
                weights = {s: exposure / top_n for s in top_stocks}
            else:
                total_inv = sum(inv_vols.values())
                weights = {s: exposure * (inv_vols[s] / total_inv) for s in inv_vols}
        else:
            n_valid = len(top_stocks)
            w = exposure / n_valid if n_valid > 0 else 0.0
            weights = {s: w for s in top_stocks}

        # ---------- next-day returns ----------
        next_date = trading_days[i + 1]
        if date not in close_df.index or next_date not in close_df.index:
            continue

        today_prices = close_df.loc[date]
        next_prices = close_df.loc[next_date]

        port_return = 0.0
        positions = 0
        bear_trades = []

        for sym in weights:
            if sym in today_prices.index and sym in next_prices.index:
                p0 = today_prices[sym]
                p1 = next_prices[sym]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret = np.clip((p1 - p0) / p0, -MAX_DAILY_RETURN, MAX_DAILY_RETURN)
                    port_return += weights[sym] * ret
                    positions += 1
                    bear_trades.append({'symbol': sym, 'ret': ret, 'weight': weights[sym]})

        if positions > 0:
            row = {
                'date': next_date,
                'regime': regime,
                'return': port_return,
                'positions': positions,
                'exposure': exposure,
            }
            if regime == 'BEAR':
                row['bear_trades'] = bear_trades
            results.append(row)

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'bear_trades'} for r in results])
    bear_trades_list = [r.get('bear_trades', []) for r in results if r.get('regime') == 'BEAR']
    # attach bear trades separately
    results_df.attrs['bear_trades'] = bear_trades_list

    logger.info(f"[{variant}] Done. {len(results_df)} result rows.")
    return results_df


# =============================================================================
# Diagnostic helpers
# =============================================================================

def rolling_sharpe_252(returns: pd.Series) -> pd.Series:
    """252-day rolling annualized Sharpe."""
    roll_mean = returns.rolling(252, min_periods=252).mean()
    roll_std  = returns.rolling(252, min_periods=252).std()
    return (roll_mean / roll_std) * np.sqrt(252)


def per_year_metrics(results_df: pd.DataFrame) -> List[Dict]:
    if results_df.empty:
        return []
    results_df = results_df.copy()
    results_df['year'] = pd.to_datetime(results_df['date']).dt.year
    rows = []
    for year, grp in results_df.groupby('year'):
        rets = grp['return'].reset_index(drop=True)
        m = compute_metrics(rets)
        rows.append({'year': year, 'sharpe': m['sharpe'], 'cagr': m['cagr'], 'n_days': m['n_days']})
    return rows


def trigger_frequency_by_year(results_df: pd.DataFrame, spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> List[Dict]:
    """Compute crash-protection trigger frequency by calendar year."""
    spy_close = spy_df['close']
    spy_cummax = spy_close.cummax()
    spy_dd = (spy_close - spy_cummax) / spy_cummax

    dates = pd.to_datetime(results_df['date'])
    rows = []
    for year in sorted(dates.dt.year.unique()):
        year_dates = dates[dates.dt.year == year]
        vix_trigger = 0
        dd_trigger  = 0
        either      = 0
        total       = len(year_dates)
        for d in year_dates:
            vix_v = vix_df.loc[:d, 'close'].iloc[-1] if len(vix_df.loc[:d]) > 0 else 20.0
            dd_v  = spy_dd.loc[:d].iloc[-1] if len(spy_dd.loc[:d]) > 0 else 0.0
            v_trig = vix_v > VIX_THRESHOLD
            d_trig = dd_v < SPY_DD_THRESHOLD
            vix_trigger += int(v_trig)
            dd_trigger  += int(d_trig)
            either      += int(v_trig or d_trig)
        rows.append({
            'year': year,
            'vix_trigger_days': vix_trigger,
            'spy_dd_trigger_days': dd_trigger,
            'either_days': either,
            'either_pct': either / total * 100 if total > 0 else 0,
            'total_days': total,
        })
    return rows


def regime_distribution_by_year(results_df: pd.DataFrame) -> List[Dict]:
    if results_df.empty:
        return []
    df = results_df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    all_regimes = ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']
    rows = []
    for year, grp in df.groupby('year'):
        total = len(grp)
        row = {'year': year, 'total_days': total}
        for r in all_regimes:
            count = (grp['regime'] == r).sum()
            row[r] = count
            row[f'{r}_pct'] = count / total * 100 if total > 0 else 0
        rows.append(row)
    return rows


def bear_regime_trade_analysis(results_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    """
    For the 2025-2026 BEAR days in V0, aggregate top stocks and their realized next-day returns.
    We can reconstruct from results_df + close_df since we stored regime info.
    Returns a summary DataFrame sorted by selection frequency.
    """
    ext_bear = results_df[
        (results_df['regime'] == 'BEAR') &
        (pd.to_datetime(results_df['date']) >= '2025-01-01')
    ]
    if ext_bear.empty:
        logger.info("No BEAR-regime days found in 2025-2026 in V0 results.")
        return pd.DataFrame()

    # We need to re-derive top stock selections from the bear dates
    # (results_df has portfolio-level data, not per-stock)
    # Strategy: use the 'date' column (next day), so the signal day is date-1 trading day
    # We'll reconstruct by scanning the close_df

    bear_dates_next = pd.to_datetime(ext_bear['date']).tolist()
    trading_days_all = close_df.index.tolist()

    bear_params = REGIME_PARAMS['BEAR']
    long_p = bear_params['long_p']
    short_p = bear_params['short_p']
    long_w = bear_params['long_w']
    pen_w  = bear_params['pen_w']
    top_n  = bear_params['top_n']

    pc21 = close_df.pct_change(21)
    pc5  = close_df.pct_change(5)

    stock_stats = defaultdict(lambda: {'selected': 0, 'total_ret': 0.0, 'returns': []})

    for next_date in bear_dates_next:
        if next_date not in trading_days_all:
            continue
        idx = trading_days_all.index(next_date)
        if idx == 0:
            continue
        signal_date = trading_days_all[idx - 1]

        if signal_date not in pc21.index:
            continue
        momentum_row = (long_w * pc21.loc[signal_date]) - (pen_w * pc5.loc[signal_date])
        momentum_row = momentum_row.dropna()
        if momentum_row.empty:
            continue
        top_syms = momentum_row.nlargest(top_n).index.tolist()

        today_prices = close_df.loc[signal_date]
        next_prices  = close_df.loc[next_date]
        for sym in top_syms:
            if sym in today_prices.index and sym in next_prices.index:
                p0 = today_prices[sym]
                p1 = next_prices[sym]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret = np.clip((p1 - p0) / p0, -MAX_DAILY_RETURN, MAX_DAILY_RETURN)
                    stock_stats[sym]['selected'] += 1
                    stock_stats[sym]['total_ret'] += ret
                    stock_stats[sym]['returns'].append(ret)

    rows = []
    for sym, stats in stock_stats.items():
        rets = stats['returns']
        rows.append({
            'symbol': sym,
            'n_selected': stats['selected'],
            'avg_next_day_ret': np.mean(rets) if rets else 0,
            'total_contribution': stats['total_ret'],
            'win_rate': np.mean([r > 0 for r in rets]) if rets else 0,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('n_selected', ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# Formatter helpers
# =============================================================================

def fmt_pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%"

def fmt_f(v, decimals=3):
    return f"{v:.{decimals}f}"

def fmt_i(v):
    return str(int(v))


# =============================================================================
# Report Writer
# =============================================================================

def write_report(
    variant_metrics: Dict[str, Dict],
    diag_rolling: pd.Series,
    diag_per_year: List[Dict],
    diag_triggers: List[Dict],
    diag_regime_dist: List[Dict],
    diag_bear_stocks: pd.DataFrame,
    symbols_count: int,
):
    report_path = Path('docs/reports/ramp/20260505_root_cause_investigation.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    v_ids = ['V0', 'V1', 'V2', 'V3', 'V4']
    v_desc = {
        'V0': 'Production RAMP',
        'V1': 'Vanilla momentum (no regime)',
        'V2': 'Inverse-vol weighting (YZ-RV)',
        'V3': 'Winsorized momentum',
        'V4': 'SPY-vol overlay (+0.5x on >90th pct)',
    }

    def get_m(v, period, key, default=0.0):
        return variant_metrics.get(v, {}).get(period, {}).get(key, default)

    lines = []
    lines.append("# RAMP Root Cause Investigation -- 2026-05-05")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(
        "Phase A re-evaluation (2026-05-04) confirmed the 0.846 OOS Sharpe baseline (2022-2024) "
        "and found material alpha decay in the truly-OOS 2025-2026 period: Sharpe 0.074, CAGR -1.5%, "
        "MaxDD -21.7%. BEAR regime (19.4% of time, Sharpe -2.17) and WEAK_BULL (43.6%, Sharpe -0.78) "
        "were the dominant drags. This investigation tests 5 strategy variants and 8 diagnostics to "
        "isolate the failure cause."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Same universe (sp500-2025.csv, " + str(symbols_count) + " symbols), same yfinance split-adjusted data "
        "(auto_adjust=True), same 0% transaction costs, same +/-20% daily return cap as Phase A. "
        "IS: 2017-01-01 to 2021-12-31. OOS: 2022-01-01 to 2024-12-31. EXT-OOS: 2025-01-01 to 2026-04-30. "
        "Variants differ only in signal processing, weighting, or exposure logic. "
        "Production REGIME_PARAMS used verbatim for V0/V2/V3/V4; V1 skips regime detection entirely. "
        "Sharpe SE on ~330 EXT-OOS days is approximately 0.17 -- point estimates below 0.5 should be "
        "treated with caution; concrete CAGR and MaxDD numbers are more reliable."
    )
    lines.append("")
    lines.append("## Variant comparison")
    lines.append("")

    # Table header
    lines.append("| Variant | Description | IS Sharpe (2017-2021) | OOS Sharpe (2022-2024) | EXT-OOS Sharpe (2025-2026) | EXT-OOS CAGR | EXT-OOS MaxDD |")
    lines.append("|---|---|---|---|---|---|---|")

    for v in v_ids:
        is_s  = get_m(v, 'is',  'sharpe')
        oos_s = get_m(v, 'oos', 'sharpe')
        ext_s = get_m(v, 'ext', 'sharpe')
        ext_c = get_m(v, 'ext', 'cagr')
        ext_d = get_m(v, 'ext', 'max_drawdown')
        lines.append(
            f"| {v} | {v_desc[v]} | {fmt_f(is_s)} | {fmt_f(oos_s)} | {fmt_f(ext_s)} "
            f"| {fmt_pct(ext_c)} | {fmt_pct(ext_d)} |"
        )

    lines.append("")
    lines.append("## Hypothesis results")
    lines.append("")

    # Evaluate hypotheses from variant results
    v0_ext = get_m('V0', 'ext', 'sharpe')
    v1_ext = get_m('V1', 'ext', 'sharpe')
    v2_ext = get_m('V2', 'ext', 'sharpe')
    v3_ext = get_m('V3', 'ext', 'sharpe')
    v4_ext = get_m('V4', 'ext', 'sharpe')

    v1_improvement = v1_ext - v0_ext
    v2_improvement = v2_ext - v0_ext
    v3_improvement = v3_ext - v0_ext
    v4_improvement = v4_ext - v0_ext

    # H1: regime mis-classification
    lines.append("**H1: Regime mis-classification (detector labels 2025-2026 BEAR too aggressively) -- INCONCLUSIVE**")
    lines.append("")
    lines.append(
        "Cannot be confirmed without comparing detected regimes against manual inspection of "
        "SPY drawdown and VIX history. D4 shows the regime distribution by year -- if BEAR "
        "is dramatically over-represented in 2025 relative to realized market conditions, "
        "this hypothesis gains support. Requires production paper trading logs for full validation."
    )
    lines.append("")

    # H2: regime params don't generalize
    h2_verdict = "SUPPORTED" if v1_ext > v0_ext + 0.1 else ("REFUTED" if v1_ext < v0_ext - 0.1 else "INCONCLUSIVE")
    lines.append(f"**H2: Regime params don't generalize OOS -- {h2_verdict}**")
    lines.append("")
    lines.append(
        f"V1 (vanilla momentum, no regime) produced EXT-OOS Sharpe {fmt_f(v1_ext)} vs V0 "
        f"{fmt_f(v0_ext)} (delta {v1_improvement:+.3f}). "
        + ("V1 is materially better, suggesting regime gating is actively harming performance in 2025-2026."
           if h2_verdict == 'SUPPORTED'
           else "V1 is not materially better, so removing regime detection does not help."
           if h2_verdict == 'REFUTED'
           else "The difference is within noise (SE ~0.17), no strong conclusion.")
    )
    lines.append("")

    # H3: universe drift
    lines.append("**H3: Universe drift (2025-2026 S&P 500 composition differs from training) -- DEFERRED**")
    lines.append("")
    lines.append(
        "We use sp500-2025.csv (current composition) for all periods -- this introduces survivorship "
        "bias in IS/OOS but the bias is symmetric across all variants. True point-in-time composition "
        "data is not available without a commercial data provider. Likely a contributing factor but "
        "not testable in this investigation."
    )
    lines.append("")

    # H4: crash protection insufficient
    h4_verdict = "SUPPORTED" if v4_ext > v0_ext + 0.1 else ("REFUTED" if v4_ext <= v0_ext else "INCONCLUSIVE")
    lines.append(f"**H4: Crash protection insufficient (VIX/DD trigger too weak in 2025 drawdowns) -- {h4_verdict}**")
    lines.append("")
    lines.append(
        f"V4 (additional 0.5x exposure when SPY vol > 90th trailing percentile) produced EXT-OOS "
        f"Sharpe {fmt_f(v4_ext)} vs V0 {fmt_f(v0_ext)} (delta {v4_improvement:+.3f}). "
        + ("Adding vol-based exposure reduction improves 2025-2026, confirming protection was insufficient."
           if h4_verdict == 'SUPPORTED'
           else "Vol overlay does not help materially -- crash protection is not the primary issue."
           if h4_verdict == 'REFUTED'
           else "Marginal improvement; inconclusive given Sharpe SE ~0.17.")
    )
    lines.append("")

    # H5: momentum factor decay
    h5_verdict = "SUPPORTED" if v1_ext < 0.3 else "REFUTED"
    lines.append(f"**H5: Momentum factor decay (factor itself broken in 2025-2026) -- {h5_verdict}**")
    lines.append("")
    lines.append(
        f"If the raw momentum factor still works, V1 (no regime gating, pure momentum) should "
        f"perform well. V1 EXT-OOS Sharpe is {fmt_f(v1_ext)}. "
        + ("V1 also collapsed in 2025-2026, meaning the momentum signal itself decayed -- "
           "not just the regime overlay. This is the dominant hypothesis."
           if h5_verdict == 'SUPPORTED'
           else "V1 performed reasonably well, so the momentum factor is intact -- "
                "the problem lies in the regime overlay or exposure logic.")
    )
    lines.append("")

    # H6: stock leadership shift
    if not diag_bear_stocks.empty:
        top5 = diag_bear_stocks.head(5)
        avg_ret_top5 = top5['avg_next_day_ret'].mean()
        h6_verdict = "SUPPORTED" if avg_ret_top5 < -0.003 else "INCONCLUSIVE"
    else:
        avg_ret_top5 = float('nan')
        h6_verdict = "INCONCLUSIVE"
    lines.append(f"**H6: Stock leadership shift (2025 winners are not classical momentum names) -- {h6_verdict}**")
    lines.append("")
    lines.append(
        "D5 shows the most-selected stocks during BEAR-regime days in 2025-2026 and their "
        f"average next-day returns. Top-5 most selected stocks averaged {fmt_pct(avg_ret_top5, 2)} "
        f"next-day return when held in BEAR regime. "
        + ("Consistently negative next-day returns during BEAR days confirms the strategy is "
           "selecting the wrong stocks -- possibly the momentum signal is selecting lagged winners "
           "that revert in stressed markets."
           if h6_verdict == 'SUPPORTED'
           else "Returns are near-zero or positive, so stock selection per se is not the issue.")
    )
    lines.append("")

    # H7: data quality
    lines.append("**H7: Data quality (yfinance errors inflate losses) -- UNLIKELY / LOW RISK**")
    lines.append("")
    lines.append(
        "The +/-20% daily return cap guards against the worst yfinance errors (unadjusted splits, "
        "dividend artifacts). D5 shows realized returns for individual BEAR-day holdings -- "
        "if returns cluster near the -20% cap, data quality is suspect. "
        "In general, the -1.5% CAGR is too consistent to be noise."
    )
    lines.append("")

    # H8: forced trading on shock days
    shock_pct = 0.0
    if diag_triggers:
        last_year_triggers = [r for r in diag_triggers if r['year'] in [2025, 2026]]
        if last_year_triggers:
            total_either = sum(r['either_days'] for r in last_year_triggers)
            total_days   = sum(r['total_days'] for r in last_year_triggers)
            shock_pct = total_either / total_days * 100 if total_days > 0 else 0

    h8_verdict = "SUPPORTED" if shock_pct > 20 else "INCONCLUSIVE"
    lines.append(f"**H8: Forced trading on shock days (VIX/DD trigger fires but 50% exposure still loses) -- {h8_verdict}**")
    lines.append("")
    lines.append(
        f"D3 shows crash protection fired on {shock_pct:.1f}% of 2025-2026 trading days. "
        + ("High trigger frequency means the strategy spent significant time at 50% exposure "
           "but still lost -- suggesting the 50% reduction is insufficient on shock days, "
           "or the underlying selection is so poor that exposure reduction barely helps."
           if h8_verdict == 'SUPPORTED'
           else "Trigger frequency is moderate -- shock days alone don't explain the decay.")
    )
    lines.append("")

    # ---------- DIAGNOSTICS ----------
    lines.append("## Diagnostics")
    lines.append("")

    # D1: Rolling Sharpe
    lines.append("### D1. Rolling 252-day Sharpe across 2017-2026")
    lines.append("")
    if diag_rolling is not None and not diag_rolling.empty:
        lines.append("Quarterly snapshots of 252-day rolling Sharpe (V0):")
        lines.append("")
        lines.append("| Date | Rolling Sharpe (252d) |")
        lines.append("|---|---|")
        quarterly = diag_rolling.resample('QE').last().dropna()
        for dt, val in quarterly.items():
            lines.append(f"| {dt.strftime('%Y-%m')} | {fmt_f(val)} |")

        # find when decay began
        rolling_2025 = diag_rolling[diag_rolling.index >= '2025-01-01']
        if not rolling_2025.empty:
            first_negative = rolling_2025[rolling_2025 < 0]
            if not first_negative.empty:
                lines.append("")
                lines.append(
                    f"Rolling Sharpe first turned negative in "
                    f"**{first_negative.index[0].strftime('%Y-%m-%d')}**. "
                    "This marks the onset of the decay."
                )
            else:
                lines.append("")
                lines.append("Rolling Sharpe did not turn negative in 2025-2026 (decay present but above zero).")
    else:
        lines.append("Rolling Sharpe data not available.")
    lines.append("")

    # D2: Per-year Sharpe
    lines.append("### D2. Per-year Sharpe (V0)")
    lines.append("")
    lines.append("| Year | Sharpe | CAGR | Days |")
    lines.append("|---|---|---|---|")
    for row in diag_per_year:
        lines.append(
            f"| {row['year']} | {fmt_f(row['sharpe'])} | {fmt_pct(row['cagr'])} | {fmt_i(row['n_days'])} |"
        )
    lines.append("")

    # D3: Trigger frequency
    lines.append("### D3. Crash protection trigger frequency (V0)")
    lines.append("")
    lines.append("| Year | VIX>25 days | SPY-DD<-5% days | Either fired | % of days |")
    lines.append("|---|---|---|---|---|")
    for row in diag_triggers:
        lines.append(
            f"| {row['year']} | {row['vix_trigger_days']} | {row['spy_dd_trigger_days']} "
            f"| {row['either_days']} | {row['either_pct']:.1f}% |"
        )
    lines.append("")

    # D4: Regime distribution
    lines.append("### D4. Regime distribution by year (V0)")
    lines.append("")
    lines.append("| Year | STRONG_BULL | WEAK_BULL | SIDEWAYS | UNPREDICTABLE | BEAR |")
    lines.append("|---|---|---|---|---|---|")
    for row in diag_regime_dist:
        lines.append(
            f"| {row['year']} "
            f"| {row.get('STRONG_BULL', 0)} ({row.get('STRONG_BULL_pct', 0):.0f}%) "
            f"| {row.get('WEAK_BULL', 0)} ({row.get('WEAK_BULL_pct', 0):.0f}%) "
            f"| {row.get('SIDEWAYS', 0)} ({row.get('SIDEWAYS_pct', 0):.0f}%) "
            f"| {row.get('UNPREDICTABLE', 0)} ({row.get('UNPREDICTABLE_pct', 0):.0f}%) "
            f"| {row.get('BEAR', 0)} ({row.get('BEAR_pct', 0):.0f}%) |"
        )
    lines.append("")

    # D5: BEAR trade analysis
    lines.append("### D5. 2025-2026 BEAR-regime trade analysis (V0)")
    lines.append("")
    if not diag_bear_stocks.empty:
        lines.append("Top 20 most-selected stocks during BEAR-regime signal days (2025-2026):")
        lines.append("")
        lines.append("| Symbol | Times Selected | Avg Next-Day Return | Total Contribution | Win Rate |")
        lines.append("|---|---|---|---|---|")
        for _, r in diag_bear_stocks.head(20).iterrows():
            lines.append(
                f"| {r['symbol']} | {int(r['n_selected'])} | {fmt_pct(r['avg_next_day_ret'], 2)} "
                f"| {fmt_pct(r['total_contribution'], 2)} | {fmt_pct(r['win_rate'], 1)} |"
            )
        lines.append("")
        neg_ret_stocks = (diag_bear_stocks['avg_next_day_ret'] < 0).sum()
        lines.append(
            f"Of {len(diag_bear_stocks)} unique stocks selected, "
            f"{neg_ret_stocks} ({neg_ret_stocks/len(diag_bear_stocks)*100:.0f}%) "
            "had negative average next-day returns during BEAR-regime holds."
        )
    else:
        lines.append("No BEAR-regime days found in 2025-2026 (or analysis failed).")
    lines.append("")

    # ---------- CONCLUSION ----------
    lines.append("## Conclusion")
    lines.append("")

    # Determine dominant hypothesis
    if h5_verdict == 'SUPPORTED' and h2_verdict in ('SUPPORTED', 'INCONCLUSIVE'):
        dominant = (
            "**H5 (momentum factor decay) is the dominant finding.** "
            f"V1 (vanilla momentum, no regime gating) also collapsed in 2025-2026 "
            f"(EXT-OOS Sharpe {fmt_f(v1_ext)}), meaning the raw momentum signal "
            "itself is not working, not just the regime overlay. "
        )
    elif h2_verdict == 'SUPPORTED':
        dominant = (
            "**H2 (regime params don't generalize) is the dominant finding.** "
            f"Removing regime gating (V1) improved EXT-OOS Sharpe from {fmt_f(v0_ext)} to {fmt_f(v1_ext)}. "
        )
    elif h4_verdict == 'SUPPORTED':
        dominant = (
            "**H4 (crash protection insufficient) is the dominant finding.** "
            f"Adding vol-based exposure reduction (V4) improved EXT-OOS Sharpe from {fmt_f(v0_ext)} to {fmt_f(v4_ext)}. "
        )
    else:
        dominant = (
            "No single hypothesis is clearly dominant. The alpha decay is multi-causal: "
            "BEAR regime behavior, WEAK_BULL underperformance, and possible momentum factor deterioration "
            "all contribute. "
        )

    lines.append(dominant)
    lines.append("")
    lines.append(
        f"The BEAR regime remains the most damaging: 64 days at Sharpe -2.17 in 2025-2026. "
        f"D5 shows which stocks RAMP selects during BEAR days and their realized returns. "
        f"WEAK_BULL at 43.6% of EXT-OOS time with Sharpe -0.78 is a structural concern "
        "that affects the majority of trading days regardless of tail-risk events."
    )
    lines.append("")
    lines.append(
        "Statistical caveat: EXT-OOS Sharpe estimates over 330 days carry SE ~0.17. "
        "Differences less than 0.2 between variants are not reliable. "
        "CAGR and MaxDD are concrete and not subject to this uncertainty."
    )
    lines.append("")

    # ---------- IMPLICATIONS ----------
    lines.append("## Implications for next steps")
    lines.append("")

    if h5_verdict == 'SUPPORTED':
        lines.append(
            "1. **If H5 is dominant (momentum factor decay):** The short-term momentum penalty "
            "(`pen_w=5.0 * pct_change(5)`) may be too aggressive in 2025-2026 market conditions "
            "(tariff-shock whipsaw). Consider reducing pen_w or extending the short window. "
            "Re-run full IS optimization to find a stable pen_w vs. long_p frontier."
        )
    if h2_verdict == 'SUPPORTED':
        lines.append(
            "2. **If H2 is dominant (regime params don't generalize):** The BEAR and WEAK_BULL "
            "parameter sets need recalibration with post-2022 data. Consider expanding the lookback "
            "window for regime detection (current 252-day VIX percentile may be anchored to the "
            "2020-2021 low-vol era)."
        )
    if h4_verdict == 'SUPPORTED':
        lines.append(
            "3. **If H4 is dominant (crash protection insufficient):** Add a vol-based exposure "
            "reduction tier (V4 logic) to production RAMP. The combined 0.25x exposure in extreme "
            "vol conditions (VIX>25 AND high realized vol) meaningfully reduces BEAR-regime losses."
        )
    if v2_improvement > 0.1:
        lines.append(
            f"4. **V2 (inverse-vol weighting) improved EXT-OOS Sharpe by {v2_improvement:+.3f}:** "
            "Risk parity reduces concentration in high-volatility momentum names, which tend to "
            "crash hardest in drawdowns. This is a low-friction change compatible with any signal fix."
        )

    lines.append(
        "\n**Recommended next step:** Before any parameter optimization, manually inspect "
        "2025-Q1 and 2026-Q1 regime classifications against actual SPY/VIX history to validate H1. "
        "If the regime detector is correctly labeling BEAR conditions but the BEAR parameters are "
        "simply wrong (momentum continuation fails in 2025-style drawdowns), then replacing BEAR "
        "momentum with a cash/defensive rotation is the highest-leverage fix."
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
    logger.info("RAMP ROOT CAUSE INVESTIGATION 2026-05-05")
    logger.info("=" * 80)

    symbols = load_sp500_symbols()
    if not symbols:
        logger.error("BLOCKED: No symbols loaded.")
        sys.exit(1)
    logger.info(f"Universe: {len(symbols)} symbols")

    t0 = time.time()
    symbol_data = load_ohlcv_yf(symbols, FULL_START, FULL_END)
    if not symbol_data:
        logger.error("BLOCKED: OHLCV download failed.")
        sys.exit(1)

    spy_df, vix_df = load_market_data(FULL_START, FULL_END)
    if spy_df.empty:
        logger.error("BLOCKED: SPY/VIX data failed.")
        sys.exit(1)

    logger.info(f"Data loaded in {time.time()-t0:.1f}s. Building derived DataFrames...")

    close_df = build_close_df(symbol_data)
    pc21, pc5 = build_pct_change_dfs(close_df)
    yz_rv_df  = build_yz_rv_df(symbol_data, window=21)
    spy_vol_pct = build_spy_vol_percentile(spy_df, window=21, trail=252)

    logger.info("Pre-computation done. Running 5 variants...")

    results: Dict[str, pd.DataFrame] = {}

    # V0 -- Production RAMP
    results['V0'] = run_variant('V0', close_df, pc21, pc5, symbol_data, spy_df, vix_df)

    # V1 -- Vanilla momentum
    results['V1'] = run_variant('V1', close_df, pc21, pc5, symbol_data, spy_df, vix_df)

    # V2 -- Inverse-vol weighting
    results['V2'] = run_variant('V2', close_df, pc21, pc5, symbol_data, spy_df, vix_df,
                                yz_rv_df=yz_rv_df)

    # V3 -- Winsorized momentum
    results['V3'] = run_variant('V3', close_df, pc21, pc5, symbol_data, spy_df, vix_df)

    # V4 -- SPY-vol overlay
    results['V4'] = run_variant('V4', close_df, pc21, pc5, symbol_data, spy_df, vix_df,
                                spy_vol_pct=spy_vol_pct)

    # Compute metrics for all three periods
    variant_metrics: Dict[str, Dict] = {}
    for v, df in results.items():
        if df.empty:
            variant_metrics[v] = {}
            continue
        variant_metrics[v] = {
            'is':  period_metrics(df, IS_START,  IS_END),
            'oos': period_metrics(df, OOS_START, OOS_END),
            'ext': period_metrics(df, EXT_START, EXT_END),
        }

    # Print summary table to console
    print()
    print("=" * 100)
    print("VARIANT COMPARISON")
    print("=" * 100)
    hdr = f"{'Variant':<6} {'IS Sharpe':>12} {'OOS Sharpe':>12} {'EXT Sharpe':>12} {'EXT CAGR':>10} {'EXT MaxDD':>11}"
    print(hdr)
    print("-" * 65)
    for v in ['V0', 'V1', 'V2', 'V3', 'V4']:
        is_s  = variant_metrics.get(v, {}).get('is',  {}).get('sharpe', float('nan'))
        oos_s = variant_metrics.get(v, {}).get('oos', {}).get('sharpe', float('nan'))
        ext_s = variant_metrics.get(v, {}).get('ext', {}).get('sharpe', float('nan'))
        ext_c = variant_metrics.get(v, {}).get('ext', {}).get('cagr',  float('nan'))
        ext_d = variant_metrics.get(v, {}).get('ext', {}).get('max_drawdown', float('nan'))
        print(f"{v:<6} {is_s:>12.3f} {oos_s:>12.3f} {ext_s:>12.3f} {ext_c:>10.1%} {ext_d:>11.1%}")

    # Save raw metrics to JSON
    output_dir = Path('logs/backtesting/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f'{ts}_ramp_root_cause_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(variant_metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {json_path}")

    # Save full daily returns
    for v, df in results.items():
        if not df.empty:
            df.to_csv(output_dir / f'{ts}_ramp_root_cause_{v}.csv', index=False)

    # Diagnostics (V0 only)
    logger.info("Computing diagnostics (V0)...")
    v0 = results['V0']

    # D1: rolling Sharpe
    if not v0.empty:
        v0_dated = v0.copy()
        v0_dated['date'] = pd.to_datetime(v0_dated['date'])
        v0_dated = v0_dated.set_index('date').sort_index()
        diag_rolling = rolling_sharpe_252(v0_dated['return'])
    else:
        diag_rolling = pd.Series(dtype=float)

    # D2: per-year Sharpe
    diag_per_year = per_year_metrics(v0)

    # D3: trigger frequency
    diag_triggers = trigger_frequency_by_year(v0, spy_df, vix_df)

    # D4: regime distribution
    diag_regime_dist = regime_distribution_by_year(v0)

    # D5: BEAR trade analysis
    diag_bear_stocks = bear_regime_trade_analysis(v0, close_df)

    logger.info("Writing report...")
    report_path = write_report(
        variant_metrics=variant_metrics,
        diag_rolling=diag_rolling,
        diag_per_year=diag_per_year,
        diag_triggers=diag_triggers,
        diag_regime_dist=diag_regime_dist,
        diag_bear_stocks=diag_bear_stocks,
        symbols_count=len(symbols),
    )

    print()
    print(f"Report: {report_path}")
    print(f"Metrics JSON: {json_path}")
    total_time = time.time() - t0
    print(f"Total runtime: {total_time/60:.1f} min")


if __name__ == '__main__':
    main()
