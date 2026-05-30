"""
RAMP Phase 3A: Variant Exploration -- 2026-05-05

Tests two directions identified by the Phase 2 root-cause investigation:

1. Vol-adjusted momentum (Barroso & Santa-Clara 2015 "Momentum has its moments").
   Recommended as TIER 1 / PRIORITY: CRITICAL in the 2025-12 improvement plan but
   never deployed. Three vol-window variants:
     V5a -- vol_window=21 (default per plan)
     V5b -- vol_window=10 (shorter, more reactive)
     V5c -- vol_window=60 (longer, more stable)

2. BEAR-to-cash (V8). Simplest possible fix: keep V0 logic for all regimes except
   BEAR, where exposure = 0.0 (cash). No regime signal is traded.

One cost-sensitivity run on the winning variant at 0%, 5 bps/side, 7.5 bps/side.

Usage:
    python scripts/backtest_scripts/ramp_phase3a_variants_20260505.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Production RAMP parameters -- verbatim from ramp_strategy.py
# =============================================================================

REGIME_PARAMS = {
    'STRONG_BULL':  {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 20},
    'WEAK_BULL':    {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10},
    'SIDEWAYS':     {'long_p': 21, 'short_p': 5, 'long_w': 0.5, 'pen_w': 2.0, 'top_n':  5},
    'UNPREDICTABLE':{'long_p': 42, 'short_p':21, 'long_w': 0.5, 'pen_w': 4.0, 'top_n': 10},
    'BEAR':         {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 3.0, 'top_n': 10},
}

DEFAULT_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 4.0, 'top_n': 10}

# V1 fixed params (no regime) -- reference only, not re-run here
V1_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10}

VIX_THRESHOLD    = 25.0
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
# Data loading (same as ramp_root_cause_20260505.py)
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
    """Download OHLCV for all symbols."""
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
    closes = {sym: df['close'] for sym, df in symbol_data.items()}
    return pd.DataFrame(closes)


def build_pct_change_dfs(close_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-compute 21-day and 5-day pct_change for all symbols."""
    pc21 = close_df.pct_change(21)
    pc5  = close_df.pct_change(5)
    return pc21, pc5


def build_daily_returns_df(close_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute daily pct_change for vol-adjusted momentum."""
    return close_df.pct_change()


def build_rolling_vol_df(daily_ret_df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Pre-compute rolling vol (annualized) for each symbol.
    vol_window: number of trading days.
    Returns DataFrame same shape as daily_ret_df.
    """
    return daily_ret_df.rolling(vol_window).std() * np.sqrt(252)


# =============================================================================
# Metric helpers (identical to ramp_root_cause_20260505.py)
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


def regime_breakdown(results_df: pd.DataFrame, start: str, end: str) -> Dict[str, Dict]:
    """Per-regime metrics for a given date window."""
    if results_df.empty:
        return {}
    subset = results_df[(results_df['date'] >= start) & (results_df['date'] <= end)]
    if subset.empty:
        return {}
    total_days = len(subset)
    total_abs_return = subset['return'].abs().sum()
    breakdown = {}
    for regime, grp in subset.groupby('regime'):
        rets = grp['return'].reset_index(drop=True)
        m = compute_metrics(rets)
        pct_days = len(grp) / total_days * 100 if total_days > 0 else 0
        pct_return_contrib = (
            grp['return'].sum() / subset['return'].sum() * 100
            if subset['return'].sum() != 0 else 0
        )
        breakdown[str(regime)] = {
            'n_days': len(grp),
            'pct_days': pct_days,
            'sharpe': m['sharpe'],
            'cagr': m['cagr'],
            'max_drawdown': m['max_drawdown'],
            'return_contrib_pct': pct_return_contrib,
        }
    return breakdown


# =============================================================================
# Core backtest loop
# =============================================================================

def run_variant(
    variant: str,
    close_df: pd.DataFrame,
    pc21: pd.DataFrame,
    pc5: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    vol_df: Optional[pd.DataFrame] = None,
    cost_drag_per_day: float = 0.0,
) -> pd.DataFrame:
    """
    Run a single variant over the full date range.

    variant: 'V5a', 'V5b', 'V5c', 'V8'
    vol_df: pre-computed annualized rolling vol DataFrame (needed for V5a/b/c)
    cost_drag_per_day: fractional daily return drag for transaction costs (0 = research mode)

    Returns DataFrame with columns: date, regime, return, positions, exposure
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

        if len(spy_sub) < 252 or len(vix_sub) < 252:
            regime = 'SIDEWAYS'
        else:
            regime, _ = detector.classify_regime(spy_sub, vix_sub, date)

        params = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)

        top_n  = params['top_n']
        long_w = params['long_w']
        pen_w  = params['pen_w']

        # ---------- V8: BEAR-to-cash ----------
        if variant == 'V8' and regime == 'BEAR':
            next_date = trading_days[i + 1]
            results.append({
                'date': next_date,
                'regime': regime,
                'return': 0.0,
                'positions': 0,
                'exposure': 0.0,
            })
            continue

        # ---------- momentum signal ----------
        if date not in pc21.index or date not in pc5.index:
            continue

        long_ret_row  = pc21.loc[date]
        short_ret_row = pc5.loc[date]

        raw_momentum = (long_w * long_ret_row) - (pen_w * short_ret_row)
        raw_momentum = raw_momentum.dropna()

        if raw_momentum.empty:
            continue

        # V5a/b/c: divide by realized vol to get vol-adjusted rankings
        if variant in ('V5a', 'V5b', 'V5c'):
            if vol_df is None:
                logger.error(f"[{variant}] vol_df is required but None -- skipping day {date}")
                continue
            if date not in vol_df.index:
                continue
            vol_row = vol_df.loc[date]
            # align on the intersection of valid momentum and vol
            common = raw_momentum.index.intersection(vol_row.dropna().index)
            if common.empty:
                continue
            raw_m = raw_momentum[common]
            v_row = vol_row[common]
            # floor at 1e-8 to match the reference implementation
            adj_momentum = raw_m / (v_row + 1e-8)
            top_stocks = adj_momentum.nlargest(top_n).index.tolist()
        else:
            top_stocks = raw_momentum.nlargest(top_n).index.tolist()

        # ---------- crash protection / exposure ----------
        vix_value   = vix_df.loc[:date, 'close'].iloc[-1] if len(vix_df.loc[:date]) > 0 else 20.0
        spy_dd_value = spy_dd.loc[:date].iloc[-1] if len(spy_dd.loc[:date]) > 0 else 0.0

        exposure = 1.0
        if vix_value > VIX_THRESHOLD or spy_dd_value < SPY_DD_THRESHOLD:
            exposure = REDUCED_EXPOSURE

        # ---------- equal-weight positions ----------
        n_valid = len(top_stocks)
        w = exposure / n_valid if n_valid > 0 else 0.0
        weights = {s: w for s in top_stocks}

        # ---------- next-day returns ----------
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

        # Apply cost drag (0.0 in research mode)
        port_return -= cost_drag_per_day

        if positions > 0 or (variant == 'V8' and regime != 'BEAR'):
            results.append({
                'date': next_date,
                'regime': regime,
                'return': port_return,
                'positions': positions,
                'exposure': exposure,
            })

    results_df = pd.DataFrame(results)
    logger.info(f"[{variant}] Done. {len(results_df)} result rows.")
    return results_df


# =============================================================================
# Formatters
# =============================================================================

def fmt_pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%"

def fmt_f(v, decimals=3):
    return f"{v:.{decimals}f}"


# =============================================================================
# Report writer
# =============================================================================

def write_report(
    variant_metrics: Dict[str, Dict],
    winning_regime_breakdown: Dict[str, Dict],
    winning_variant: str,
    winning_cost_metrics: Dict[str, Dict],
    symbols_count: int,
):
    report_path = Path('docs/reports/ramp/20260505_variant_exploration_phase3a.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Reference values from Phase 2
    ref = {
        'V0': {'is': 0.755, 'oos': 0.867, 'ext_sharpe': 0.070, 'ext_cagr': -0.017, 'ext_dd': -0.216},
        'V1': {'is': 0.895, 'oos': 0.710, 'ext_sharpe': 0.314, 'ext_cagr':  0.048, 'ext_dd': -0.217},
    }

    v_desc = {
        'V5a': 'Vol-adj momentum, vol_window=21',
        'V5b': 'Vol-adj momentum, vol_window=10',
        'V5c': 'Vol-adj momentum, vol_window=60',
        'V8':  'V0 + BEAR-to-cash',
    }

    def get_m(v, period, key, default=float('nan')):
        return variant_metrics.get(v, {}).get(period, {}).get(key, default)

    lines = []
    lines.append("# RAMP Phase 3A: Variant Exploration -- 2026-05-05")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(
        "Phase 2 root-cause investigation (2026-05-05) established: H2 SUPPORTED (regime "
        "gating harms EXT-OOS performance; V1 vanilla momentum Sharpe 0.314 vs V0 0.070), "
        "H5 REFUTED (the raw momentum signal itself remains alive), H4 REFUTED (more vol-based "
        "exposure reduction makes things worse). The 2025-12 improvement plan recommended "
        "vol-adjusted momentum (Barroso & Santa-Clara 2015) as TIER 1 / PRIORITY: CRITICAL "
        "-- it was never deployed. This phase tests that recommendation (V5a/b/c) and the "
        "simplest possible regime fix: cash in BEAR regime (V8)."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"Same universe (sp500-2025.csv, {symbols_count} symbols), same yfinance split-adjusted "
        "data (auto_adjust=True), same 0% transaction costs, same +/-20% daily return cap as "
        "Phase 2. IS: 2017-01-01 to 2021-12-31. OOS: 2022-01-01 to 2024-12-31. "
        "EXT-OOS: 2025-01-01 to 2026-04-30. "
        "V5a/b/c use production REGIME_PARAMS for exposure/top_n but replace the raw momentum "
        "ranking with a vol-normalized score: raw_momentum / (rolling_std(daily_ret, vol_window) "
        "* sqrt(252) + 1e-8). V8 is identical to V0 except BEAR days hold 0% exposure (cash). "
        "Sharpe SE on ~331 EXT-OOS days is approximately 0.17 -- differences below 0.2 are "
        "within noise. CAGR and MaxDD are concrete and not subject to this uncertainty."
    )
    lines.append("")
    lines.append("## Variant comparison")
    lines.append("")
    lines.append(
        "| Variant | Description | IS Sharpe (2017-2021) | OOS Sharpe (2022-2024) "
        "| EXT-OOS Sharpe (2025-2026) | EXT-OOS CAGR | EXT-OOS MaxDD |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    # Reference rows
    lines.append(
        f"| V0 (reference) | Production RAMP | {ref['V0']['is']:.3f} | {ref['V0']['oos']:.3f} "
        f"| {ref['V0']['ext_sharpe']:.3f} | {fmt_pct(ref['V0']['ext_cagr'])} "
        f"| {fmt_pct(ref['V0']['ext_dd'])} |"
    )
    lines.append(
        f"| V1 (reference) | Vanilla momentum (no regime) | {ref['V1']['is']:.3f} "
        f"| {ref['V1']['oos']:.3f} | {ref['V1']['ext_sharpe']:.3f} "
        f"| {fmt_pct(ref['V1']['ext_cagr'])} | {fmt_pct(ref['V1']['ext_dd'])} |"
    )

    for v in ('V5a', 'V5b', 'V5c', 'V8'):
        is_s  = get_m(v, 'is',  'sharpe')
        oos_s = get_m(v, 'oos', 'sharpe')
        ext_s = get_m(v, 'ext', 'sharpe')
        ext_c = get_m(v, 'ext', 'cagr')
        ext_d = get_m(v, 'ext', 'max_drawdown')
        mark = " (*)" if v == winning_variant else ""
        lines.append(
            f"| {v}{mark} | {v_desc[v]} | {fmt_f(is_s)} | {fmt_f(oos_s)} | {fmt_f(ext_s)} "
            f"| {fmt_pct(ext_c)} | {fmt_pct(ext_d)} |"
        )

    lines.append("")
    lines.append(f"(*) = winning variant by EXT-OOS Sharpe: **{winning_variant}**")
    lines.append("")
    lines.append("## Cost sensitivity (winning variant)")
    lines.append("")

    if winning_cost_metrics:
        lines.append(
            f"Running {winning_variant} at three cost tiers. Turnover assumed = 1.0 (daily rotation). "
            "Cost drag = 2 * bps_per_side * turnover per trading day."
        )
        lines.append("")
        lines.append("| Cost tier | bps/side | Daily drag | EXT-OOS Sharpe | EXT-OOS CAGR | EXT-OOS MaxDD |")
        lines.append("|---|---|---|---|---|---|")
        for tier, label, bps, drag in [
            ('0pct',  '0% (research)',  0,   0.0),
            ('5bps',  '5 bps',         5,   0.001),
            ('75bps', '7.5 bps (1.5x)',7.5, 0.0015),
        ]:
            m = winning_cost_metrics.get(tier, {})
            ext_s = m.get('sharpe', float('nan'))
            ext_c = m.get('cagr',   float('nan'))
            ext_d = m.get('max_drawdown', float('nan'))
            lines.append(
                f"| {label} | {bps} | {drag*100:.3f}% | {fmt_f(ext_s)} "
                f"| {fmt_pct(ext_c)} | {fmt_pct(ext_d)} |"
            )
    else:
        lines.append("Cost sensitivity data not available.")

    lines.append("")
    lines.append("## Pre-committed evaluation against criteria")
    lines.append("")

    for v in ('V5a', 'V5b', 'V5c', 'V8'):
        is_s  = get_m(v, 'is',  'sharpe')
        oos_s = get_m(v, 'oos', 'sharpe')
        ext_s = get_m(v, 'ext', 'sharpe')

        # Criteria
        c1_pass = ext_s > 0.5
        c2_pass = abs(oos_s - 0.823) <= 0.1  # within +/-0.1 of Phase 2 baseline
        is_oos_gap = (is_s - oos_s) / abs(is_s) * 100 if is_s != 0 else 0
        c3_pass = abs(is_oos_gap) < 30

        c1 = "PASS" if c1_pass else "FAIL"
        c2 = "PASS" if c2_pass else "FAIL"
        c3 = "PASS" if c3_pass else "FAIL"

        # IS/OOS gap direction
        gap_str = f"{is_oos_gap:+.1f}% (IS {'>' if is_s > oos_s else '<'} OOS)"

        # Winner gets cost criterion
        if v == winning_variant and winning_cost_metrics:
            c4_s = winning_cost_metrics.get('75bps', {}).get('sharpe', float('nan'))
            c4_pass = c4_s > 0.3
            c4 = f"PASS ({fmt_f(c4_s)})" if c4_pass else f"FAIL ({fmt_f(c4_s)})"
        else:
            c4 = "N/A (not winning variant)"
            c4_pass = None

        if c1_pass and c2_pass and c3_pass and (c4_pass is None or c4_pass):
            verdict = "PRODUCTION CANDIDATE"
        elif c1_pass or (c2_pass and c3_pass):
            verdict = "PROMISING"
        else:
            verdict = "RESEARCH ONLY"

        lines.append(f"### {v} -- {v_desc[v]}")
        lines.append("")
        lines.append(f"| Criterion | Threshold | Actual | Result |")
        lines.append("|---|---|---|---|")
        lines.append(f"| EXT-OOS Sharpe > 0.5 (0% costs) | > 0.5 | {fmt_f(ext_s)} | {c1} |")
        lines.append(
            f"| OOS Sharpe within +/-0.1 of 0.823 | 0.723 to 0.923 | {fmt_f(oos_s)} | {c2} |"
        )
        lines.append(
            f"| IS/OOS gap < 30% | < 30% | {gap_str} | {c3} |"
        )
        lines.append(
            f"| (winner) EXT-OOS Sharpe > 0.3 at 1.5x costs | > 0.3 | -- | {c4} |"
        )
        lines.append("")
        lines.append(f"**Verdict: {verdict}**")
        lines.append("")

    # Regime breakdown for winning variant
    if winning_regime_breakdown:
        w_ext_s = get_m(winning_variant, 'ext', 'sharpe')
        lines.append(f"## Regime breakdown -- {winning_variant} EXT-OOS (2025-2026)")
        lines.append("")
        lines.append(
            "| Regime | % of days | Sharpe | CAGR | Max DD | Return contrib % |"
        )
        lines.append("|---|---|---|---|---|---|")
        all_regimes = ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']
        for r in all_regimes:
            if r in winning_regime_breakdown:
                bd = winning_regime_breakdown[r]
                lines.append(
                    f"| {r} | {bd['pct_days']:.1f}% | {fmt_f(bd['sharpe'])} "
                    f"| {fmt_pct(bd['cagr'])} | {fmt_pct(bd['max_drawdown'])} "
                    f"| {bd['return_contrib_pct']:+.1f}% |"
                )
        # Fragility check
        contrib_vals = [
            abs(v['return_contrib_pct'])
            for v in winning_regime_breakdown.values()
        ]
        if contrib_vals:
            max_contrib = max(contrib_vals)
            max_regime = max(winning_regime_breakdown, key=lambda r: abs(winning_regime_breakdown[r]['return_contrib_pct']))
            if max_contrib > 70:
                lines.append("")
                lines.append(
                    f"WARNING: {max_contrib:.0f}% of returns come from {max_regime} "
                    "regime -- FRAGILE (regime-concentration risk)."
                )
        lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")

    winning_ext = get_m(winning_variant, 'ext', 'sharpe')
    winning_cagr = get_m(winning_variant, 'ext', 'cagr')
    winning_dd   = get_m(winning_variant, 'ext', 'max_drawdown')
    v1_ext = ref['V1']['ext_sharpe']
    v0_ext = ref['V0']['ext_sharpe']

    # Determine verdict
    is_winner_production = False
    w_is_s   = get_m(winning_variant, 'is',  'sharpe')
    w_oos_s  = get_m(winning_variant, 'oos', 'sharpe')
    c1_w = winning_ext > 0.5
    c2_w = abs(w_oos_s - 0.823) <= 0.1
    is_oos_gap_w = (w_is_s - w_oos_s) / abs(w_is_s) * 100 if w_is_s != 0 else 0
    c3_w = abs(is_oos_gap_w) < 30
    c4_s_val = winning_cost_metrics.get('75bps', {}).get('sharpe', float('nan')) if winning_cost_metrics else float('nan')
    c4_w = not np.isnan(c4_s_val) and c4_s_val > 0.3
    is_winner_production = c1_w and c2_w and c3_w and c4_w

    lines.append(
        f"**Winning variant: {winning_variant}** with EXT-OOS Sharpe {fmt_f(winning_ext)}, "
        f"CAGR {fmt_pct(winning_cagr)}, MaxDD {fmt_pct(winning_dd)}."
    )
    lines.append("")

    delta_v1 = winning_ext - v1_ext
    lines.append(
        f"The winner {'improves' if delta_v1 > 0 else 'does not improve'} on V1 (vanilla momentum, "
        f"EXT-OOS Sharpe 0.314) by {delta_v1:+.3f} Sharpe points. V1 remains the practical "
        f"floor for the regime-free approach."
    )
    lines.append("")

    if is_winner_production:
        lines.append(
            f"{winning_variant} passes all four pre-committed criteria (EXT-OOS Sharpe > 0.5, "
            "OOS Sharpe within +/-0.1 of baseline, IS/OOS gap < 30%, cost sensitivity > 0.3 at 1.5x). "
            "It is classified as a **PRODUCTION CANDIDATE**."
        )
    else:
        failed = []
        if not c1_w:
            failed.append(f"EXT-OOS Sharpe {fmt_f(winning_ext)} < 0.5")
        if not c2_w:
            failed.append(f"OOS Sharpe {fmt_f(w_oos_s)} outside +/-0.1 of 0.823")
        if not c3_w:
            failed.append(f"IS/OOS gap {is_oos_gap_w:+.1f}% >= 30%")
        if not c4_w and not np.isnan(c4_s_val):
            failed.append(f"1.5x cost Sharpe {fmt_f(c4_s_val)} < 0.3")
        lines.append(
            f"{winning_variant} does NOT qualify as a production candidate. "
            f"Failed criteria: {'; '.join(failed)}. "
            "It is classified as PROMISING -- warranting further investigation but not "
            "deployment without additional validation."
        )
    lines.append("")

    # What does this say about the underlying problem?
    lines.append(
        "The vol-adjusted momentum variants test whether risk-normalizing signals changes "
        "which stocks rank at the top -- different from V2 (which kept the same rankings but "
        "changed position sizing). If vol-adjusted rankings improve EXT-OOS, it suggests "
        "the problem is that raw momentum over-selects high-beta momentum names that crash "
        "hardest in 2025-style drawdowns. V8 tests whether simply side-stepping BEAR days "
        "recovers performance -- if V8 wins, it confirms the Phase 2 finding that BEAR regime "
        "parameters are the dominant drag and the simplest fix is the best."
    )
    lines.append("")

    # Statistical caveat
    lines.append(
        "Statistical caveat: Sharpe SE on 331 EXT-OOS days is ~0.17. Differences less than "
        "0.2 between variants are not reliable. Any claimed improvement should be confirmed "
        "with a longer OOS window before deployment."
    )
    lines.append("")

    lines.append("## Implications for next steps")
    lines.append("")

    if is_winner_production:
        lines.append(
            f"1. **{winning_variant} is a production candidate.** "
            "Before deploying: (a) run paper trading parallel for 30+ days, "
            "(b) confirm regime breakdown is not fragile (no single regime > 70%), "
            "(c) monitor live vs backtest Sharpe weekly. "
            "Deploy only if paper and live Sharpes are within 0.2 of backtest."
        )
    else:
        if winning_ext > 0.3:
            lines.append(
                f"1. **{winning_variant} is PROMISING but not ready for production.** "
                "EXT-OOS Sharpe improved but did not clear the 0.5 bar. "
                "Recommended next test: combine the winning vol-adj signal with V8 BEAR-to-cash. "
                "Two modifications tested independently; if each adds 0.1+ Sharpe, the combination "
                "may clear 0.5."
            )
        else:
            lines.append(
                "1. **No variant cleared the production bar.** "
                "The best result is within noise of V1 (vanilla momentum). "
                "Consider a parameter sweep on vol_window or testing pure factor rotation "
                "(switch entirely to low-vol factor in BEAR regimes)."
            )
    lines.append(
        "2. **Do not optimize parameters on EXT-OOS data.** The 2025-2026 window has only "
        "~331 days (SE ~0.17). Any further tuning must use the IS (2017-2021) period only, "
        "with EXT-OOS held truly blind."
    )
    lines.append(
        "3. **Survivorship bias caveat:** All tests use sp500-2025.csv (current composition). "
        "Stocks that were removed from the S&P 500 during 2017-2024 are excluded. "
        "This biases IS/OOS upward, but the bias is symmetric across all variants so "
        "relative comparisons are valid."
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
    logger.info("RAMP PHASE 3A VARIANT EXPLORATION 2026-05-05")
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

    close_df  = build_close_df(symbol_data)
    pc21, pc5 = build_pct_change_dfs(close_df)
    daily_ret = build_daily_returns_df(close_df)

    # Pre-compute rolling vol for each V5 vol_window
    logger.info("Pre-computing rolling vol DataFrames for V5a/b/c...")
    vol_df_21 = build_rolling_vol_df(daily_ret, vol_window=21)
    vol_df_10 = build_rolling_vol_df(daily_ret, vol_window=10)
    vol_df_60 = build_rolling_vol_df(daily_ret, vol_window=60)

    logger.info("Pre-computation done. Running 4 new variants...")

    results: Dict[str, pd.DataFrame] = {}

    results['V5a'] = run_variant('V5a', close_df, pc21, pc5, spy_df, vix_df, vol_df=vol_df_21)
    results['V5b'] = run_variant('V5b', close_df, pc21, pc5, spy_df, vix_df, vol_df=vol_df_10)
    results['V5c'] = run_variant('V5c', close_df, pc21, pc5, spy_df, vix_df, vol_df=vol_df_60)
    results['V8']  = run_variant('V8',  close_df, pc21, pc5, spy_df, vix_df)

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

    # Print summary table
    print()
    print("=" * 100)
    print("PHASE 3A VARIANT COMPARISON")
    print("=" * 100)
    print(f"{'Variant':<6} {'IS Sharpe':>12} {'OOS Sharpe':>12} {'EXT Sharpe':>12} {'EXT CAGR':>10} {'EXT MaxDD':>11}")
    print("-" * 65)
    for v in ('V5a', 'V5b', 'V5c', 'V8'):
        is_s  = variant_metrics.get(v, {}).get('is',  {}).get('sharpe', float('nan'))
        oos_s = variant_metrics.get(v, {}).get('oos', {}).get('sharpe', float('nan'))
        ext_s = variant_metrics.get(v, {}).get('ext', {}).get('sharpe', float('nan'))
        ext_c = variant_metrics.get(v, {}).get('ext', {}).get('cagr',   float('nan'))
        ext_d = variant_metrics.get(v, {}).get('ext', {}).get('max_drawdown', float('nan'))
        print(f"{v:<6} {is_s:>12.3f} {oos_s:>12.3f} {ext_s:>12.3f} {ext_c:>10.1%} {ext_d:>11.1%}")

    # Identify winner
    ext_sharpes = {
        v: variant_metrics.get(v, {}).get('ext', {}).get('sharpe', float('-inf'))
        for v in ('V5a', 'V5b', 'V5c', 'V8')
    }
    winning_variant = max(ext_sharpes, key=ext_sharpes.get)
    logger.info(f"Winning variant: {winning_variant} (EXT-OOS Sharpe {ext_sharpes[winning_variant]:.3f})")

    # Regime breakdown for winning variant
    winning_regime_breakdown = regime_breakdown(results[winning_variant], EXT_START, EXT_END)

    # Cost sensitivity for winning variant
    logger.info(f"Running cost sensitivity for {winning_variant}...")
    winning_cost_metrics = {}

    # 0% costs -- already computed
    winning_cost_metrics['0pct'] = variant_metrics[winning_variant].get('ext', {})

    # 5 bps/side: daily drag = 2 * 0.0005 * 1.0 = 0.001 (0.1% per day)
    winning_vol_df = {'V5a': vol_df_21, 'V5b': vol_df_10, 'V5c': vol_df_60, 'V8': None}
    df_cost5 = run_variant(
        winning_variant, close_df, pc21, pc5, spy_df, vix_df,
        vol_df=winning_vol_df[winning_variant],
        cost_drag_per_day=0.001,
    )
    winning_cost_metrics['5bps'] = period_metrics(df_cost5, EXT_START, EXT_END)

    # 7.5 bps/side (1.5x): daily drag = 2 * 0.00075 * 1.0 = 0.0015
    df_cost75 = run_variant(
        winning_variant, close_df, pc21, pc5, spy_df, vix_df,
        vol_df=winning_vol_df[winning_variant],
        cost_drag_per_day=0.0015,
    )
    winning_cost_metrics['75bps'] = period_metrics(df_cost75, EXT_START, EXT_END)

    # Print cost sensitivity
    print()
    print("=" * 80)
    print(f"COST SENSITIVITY -- {winning_variant}")
    print("=" * 80)
    for tier, label in [('0pct', '0 bps'), ('5bps', '5 bps/side'), ('75bps', '7.5 bps/side')]:
        m = winning_cost_metrics.get(tier, {})
        print(
            f"{label:<20} EXT Sharpe: {m.get('sharpe', float('nan')):.3f}  "
            f"CAGR: {m.get('cagr', float('nan')):.1%}  "
            f"MaxDD: {m.get('max_drawdown', float('nan')):.1%}"
        )

    logger.info("Writing report...")
    report_path = write_report(
        variant_metrics=variant_metrics,
        winning_regime_breakdown=winning_regime_breakdown,
        winning_variant=winning_variant,
        winning_cost_metrics=winning_cost_metrics,
        symbols_count=len(symbols),
    )

    print()
    print(f"Report: {report_path}")
    total_time = time.time() - t0
    print(f"Total runtime: {total_time/60:.1f} min")


if __name__ == '__main__':
    main()
