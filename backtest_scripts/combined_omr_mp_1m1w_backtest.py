"""
Combined OMR + MP (1m-1w) Backtest.

Tests OMR with production Bayesian model + MP with 1 month - 1 week momentum.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
from src.utils.logger import logger
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel

# ============================================================================
# Configuration
# ============================================================================
OMR_CONFIG = {
    'position_size': 0.15,
    'stop_loss': -0.02,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'vix_threshold': 35,
    'max_positions': 3,
    'skip_regimes': ['BEAR'],
    'symbols': ['FAZ', 'UDOW', 'SOXL', 'TECL', 'UPRO', 'SVXY', 'TQQQ', 'SSO',
                'DFEN', 'LABU', 'TNA', 'SQQQ', 'ERX', 'RETL']
}

MP_CONFIG = {
    'position_size': 0.065,
    'top_n': 10,
    'vix_threshold': 25.0,
    'spy_dd_threshold': -0.05,
}

DATA_DIR = PROJECT_ROOT / 'data' / 'leveraged_etfs'


def calc_metrics(returns, name):
    """Calculate performance metrics."""
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / vol if vol > 0 else 0
    rolling_max = cum.expanding().max()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()
    return {
        'name': name,
        'cum': total_ret,
        'annual': annual_ret,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


def main():
    print('='*70)
    print(' COMBINED OMR + MP (1m-1w) BACKTEST')
    print('='*70)
    print('Period: 2017-01-01 to 2024-12-31')
    print()
    print('Configuration:')
    print('  OMR: Production Bayesian model, 15% per position, max 3')
    print('  MP:  1 month - 1 week momentum, 6.5% per position, top 10')
    print()

    # ========================================================================
    # Load OMR Data
    # ========================================================================
    print('Loading OMR data from parquet files...')
    omr_data = {}

    spy_df = pd.read_parquet(DATA_DIR / 'SPY_1d.parquet')
    vix_df = pd.read_parquet(DATA_DIR / '^VIX_1d.parquet')

    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [col[0] for col in spy_df.columns]
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [col[0] for col in vix_df.columns]

    spy_df.columns = [col.lower() for col in spy_df.columns]
    vix_df.columns = [col.lower() for col in vix_df.columns]
    spy_df.index = pd.to_datetime(spy_df.index)
    vix_df.index = pd.to_datetime(vix_df.index)

    omr_data['SPY'] = spy_df
    omr_data['^VIX'] = vix_df

    for symbol in OMR_CONFIG['symbols']:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.columns = [col.lower() for col in df.columns]
            df.index = pd.to_datetime(df.index)
            omr_data[symbol] = df

    print(f'  Loaded {len(omr_data)-2} OMR symbols')

    # ========================================================================
    # Load MP Data
    # ========================================================================
    print('Downloading MP data...')
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'
    mp_symbols = pd.read_csv(csv_path)['Symbol'].tolist()

    mp_data = yf.download(mp_symbols + ['SPY', '^VIX'], start='2016-01-01',
                          end='2024-12-31', progress=False, auto_adjust=True)
    mp_prices = mp_data['Close'] if isinstance(mp_data.columns, pd.MultiIndex) else mp_data
    mp_spy = mp_prices['SPY']
    mp_vix = mp_prices['^VIX']

    print(f'  {len(mp_prices)} trading days, {len(mp_symbols)} symbols')
    print()

    # ========================================================================
    # Run OMR Backtest
    # ========================================================================
    print('Running OMR backtest with production Bayesian model...')

    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel()
    bayesian_model.load_model()

    spy_data = omr_data['SPY']
    vix_data = omr_data['^VIX']

    test_dates = spy_data.loc['2017-01-01':'2024-12-31'].index
    omr_daily = []

    for date in test_dates:
        if date not in spy_data.index or date not in vix_data.index:
            continue

        regime, _ = regime_detector.classify_regime(spy_data, vix_data, date)

        if regime in OMR_CONFIG['skip_regimes']:
            omr_daily.append({'date': date, 'return': 0.0})
            continue

        vix_val = float(vix_data[vix_data.index <= date]['close'].iloc[-1])
        if vix_val > OMR_CONFIG['vix_threshold']:
            omr_daily.append({'date': date, 'return': 0.0})
            continue

        day_trades = []
        for symbol in OMR_CONFIG['symbols']:
            if symbol not in omr_data:
                continue

            sym_data = omr_data[symbol]
            if date not in sym_data.index:
                continue

            today = sym_data.loc[date]
            today_open = float(today['open'] if isinstance(today['open'], (int, float)) else today['open'].iloc[0])
            today_close = float(today['close'] if isinstance(today['close'], (int, float)) else today['close'].iloc[0])

            intraday_return = (today_close - today_open) / today_open

            if abs(intraday_return) < 0.005:
                continue

            prob_data = bayesian_model.get_reversion_probability(symbol, regime, intraday_return)

            if prob_data is None:
                continue

            if (prob_data['probability'] < OMR_CONFIG['min_win_rate'] or
                prob_data['expected_return'] < OMR_CONFIG['min_expected_return'] or
                prob_data['sample_size'] < OMR_CONFIG['min_sample_size']):
                continue

            next_idx = sym_data.index.get_loc(date) + 1
            if next_idx >= len(sym_data):
                continue

            next_open = float(sym_data.iloc[next_idx]['open'])
            overnight_return = (next_open - today_close) / today_close

            if overnight_return < OMR_CONFIG['stop_loss']:
                overnight_return = OMR_CONFIG['stop_loss']

            day_trades.append({
                'symbol': symbol,
                'overnight_return': overnight_return,
                'probability': prob_data['probability']
            })

        if day_trades:
            day_trades.sort(key=lambda x: x['probability'], reverse=True)
            selected = day_trades[:OMR_CONFIG['max_positions']]
            port_ret = sum(t['overnight_return'] * OMR_CONFIG['position_size'] for t in selected)
            omr_daily.append({'date': date, 'return': port_ret})
        else:
            omr_daily.append({'date': date, 'return': 0.0})

    omr_returns = pd.DataFrame(omr_daily).set_index('date')['return']
    print(f'  OMR: {len(omr_returns)} days')

    # ========================================================================
    # Run MP (1m-1w) Backtest
    # ========================================================================
    print('Running MP backtest with 1m-1w momentum...')

    # 1m-1w momentum
    returns_1m = mp_prices.pct_change(21)
    returns_1w = mp_prices.pct_change(5)
    momentum_all = returns_1m - returns_1w

    high_vix = mp_vix > MP_CONFIG['vix_threshold']
    spy_rolling_max = mp_spy.expanding().max()
    spy_drawdown = (mp_spy - spy_rolling_max) / spy_rolling_max
    spy_dd_trigger = spy_drawdown < MP_CONFIG['spy_dd_threshold']

    daily_returns = mp_prices.pct_change()
    mp_test_dates = mp_prices.loc['2017-01-01':'2024-12-31'].index
    mp_daily = []

    for date in mp_test_dates:
        if date not in momentum_all.index:
            continue

        mom_today = momentum_all.loc[date].dropna()

        if len(mom_today) < MP_CONFIG['top_n']:
            mp_daily.append({'date': date, 'return': 0.0})
            continue

        risk_active = False
        if date in high_vix.index and high_vix.loc[date]:
            risk_active = True
        if date in spy_dd_trigger.index and spy_dd_trigger.loc[date]:
            risk_active = True

        exposure = 0.5 if risk_active else 1.0

        top_stocks = mom_today.nlargest(MP_CONFIG['top_n']).index.tolist()
        valid_stocks = [s for s in top_stocks if s in daily_returns.columns and date in daily_returns.index]

        if not valid_stocks:
            mp_daily.append({'date': date, 'return': 0.0})
            continue

        port_return = 0.0
        for stock in valid_stocks:
            stock_ret = daily_returns.loc[date, stock]
            if pd.notna(stock_ret):
                port_return += stock_ret * MP_CONFIG['position_size'] * exposure

        mp_daily.append({'date': date, 'return': port_return})

    mp_returns = pd.DataFrame(mp_daily).set_index('date')['return']
    print(f'  MP:  {len(mp_returns)} days')

    # ========================================================================
    # Combine Strategies
    # ========================================================================
    print()
    print('Combining strategies...')

    common_dates = omr_returns.index.intersection(mp_returns.index)
    omr_aligned = omr_returns.loc[common_dates]
    mp_aligned = mp_returns.loc[common_dates]
    combined = omr_aligned + mp_aligned

    # SPY benchmark
    spy_bench = mp_spy.pct_change().loc[common_dates]

    print(f'  Combined: {len(combined)} days')
    print()

    # ========================================================================
    # Calculate Metrics
    # ========================================================================
    omr_m = calc_metrics(omr_aligned, 'OMR')
    mp_m = calc_metrics(mp_aligned, 'MP (1m-1w)')
    comb_m = calc_metrics(combined, 'Combined')
    spy_m = calc_metrics(spy_bench, 'SPY')

    print('='*70)
    print(' RESULTS')
    print('='*70)
    print()

    for m in [omr_m, mp_m, comb_m, spy_m]:
        print(f"{m['name']}:")
        print(f"  Cumulative Return: {m['cum']*100:+.1f}%")
        print(f"  Annual Return:     {m['annual']*100:+.1f}%")
        print(f"  Annual Volatility: {m['vol']*100:.1f}%")
        print(f"  Sharpe Ratio:      {m['sharpe']:.2f}")
        print(f"  Max Drawdown:      {m['max_dd']*100:.1f}%")
        print()

    # ========================================================================
    # Yearly Breakdown
    # ========================================================================
    print('='*70)
    print(' YEARLY BREAKDOWN')
    print('='*70)
    print()
    print(f"{'Year':>6}{'OMR':>12}{'MP(1m-1w)':>12}{'Combined':>12}{'SPY':>12}")
    print('-'*54)

    omr_yearly = omr_aligned.resample('Y').apply(lambda x: (1+x).prod()-1)
    mp_yearly = mp_aligned.resample('Y').apply(lambda x: (1+x).prod()-1)
    comb_yearly = combined.resample('Y').apply(lambda x: (1+x).prod()-1)
    spy_yearly = spy_bench.resample('Y').apply(lambda x: (1+x).prod()-1)

    for yr in omr_yearly.index:
        print(f"{yr.year:>6}{omr_yearly.loc[yr]*100:>+11.1f}%{mp_yearly.loc[yr]*100:>+11.1f}%{comb_yearly.loc[yr]*100:>+11.1f}%{spy_yearly.loc[yr]*100:>+11.1f}%")

    # ========================================================================
    # Monthly Stats
    # ========================================================================
    print()
    print('='*70)
    print(' MONTHLY STATISTICS')
    print('='*70)
    print()

    for name, rets in [('OMR', omr_aligned), ('MP (1m-1w)', mp_aligned), ('Combined', combined)]:
        monthly = rets.resample('M').apply(lambda x: (1+x).prod()-1)
        pos = (monthly > 0).sum()
        total = len(monthly)
        avg_up = monthly[monthly>0].mean() * 100 if (monthly>0).any() else 0
        avg_down = monthly[monthly<=0].mean() * 100 if (monthly<=0).any() else 0
        print(f"{name}:")
        print(f"  Win Rate: {pos/total*100:.1f}% ({pos}/{total} months)")
        print(f"  Avg Up:   {avg_up:+.2f}%")
        print(f"  Avg Down: {avg_down:+.2f}%")
        print()


if __name__ == '__main__':
    main()
