"""
Multi-Strategy (OMR + MP) Backtest with Monthly Contributions.

Tests combined strategy performance with dollar-cost averaging.

MP Strategy Timing: 3:55 PM ET (NO SHIFTS NEEDED)
- Each day: measure return for YESTERDAY's positions
- Then: select NEW positions using today's momentum
- Loop structure naturally handles the timing - no shift() calls

Timeline:
  Day T (3:55 PM): Select stocks based on momentum[T] → hold overnight
  Day T+1 (3:55 PM): Measure return (close[T+1]/close[T]-1), select new stocks

This avoids lookahead bias because:
- Returns are measured for PREVIOUS day's selections
- New selections use CURRENT data (known at decision time)
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
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel

# Configuration
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
    'slippage_per_share': 0.02,  # $0.02 per share slippage
    'avg_stock_price': 150,      # Assume avg S&P 500 stock price ~$150
}

INITIAL_CAPITAL = 50000
MONTHLY_CONTRIBUTION = 1000
DATA_DIR = PROJECT_ROOT / 'data' / 'leveraged_etfs'


def main():
    print('='*70)
    print(' MULTI-STRATEGY (OMR + MP) WITH MONTHLY CONTRIBUTIONS')
    print('='*70)
    print()
    print(f'Initial Investment: ${INITIAL_CAPITAL:,}')
    print(f'Monthly Contribution: ${MONTHLY_CONTRIBUTION:,}')
    print('Period: 2017-01-01 to 2024-12-31')
    print()
    print('Strategies:')
    print('  OMR: Bayesian overnight reversion (15% x 3 positions = 45%)')
    print('       Entry: 3:50 PM, Exit: 9:35 AM next day')
    print('  MP:  1m-1w momentum (6.5% x 10 positions = 65%)')
    print('       Entry: 3:55 PM, Exit: 3:55 PM next day (close-to-close)')
    print('  Combined exposure: ~110% (slight leverage)')
    print()

    # Load OMR data
    print('Loading OMR data...')
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

    # Load MP data
    print('Loading MP data...')
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'
    mp_symbols = pd.read_csv(csv_path)['Symbol'].tolist()

    mp_raw = yf.download(mp_symbols + ['SPY', '^VIX'], start='2016-01-01',
                         end='2024-12-31', progress=False, auto_adjust=True)
    mp_prices = mp_raw['Close'] if isinstance(mp_raw.columns, pd.MultiIndex) else mp_raw
    mp_spy = mp_prices['SPY']
    mp_vix = mp_prices['^VIX']

    # Initialize OMR models
    print('Loading Bayesian model...')
    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel()
    bayesian_model.load_model()

    # Pre-compute MP signals (1m-1w momentum)
    # MP runs at 3:55 PM - use TODAY's close to select stocks
    # NO SHIFTS NEEDED - loop structure handles timing naturally
    returns_1m = mp_prices.pct_change(21)
    returns_1w = mp_prices.pct_change(5)
    mp_momentum = returns_1m - returns_1w  # Today's momentum

    mp_high_vix = mp_vix > MP_CONFIG['vix_threshold']  # Today's VIX

    mp_spy_max = mp_spy.expanding().max()
    mp_spy_dd = (mp_spy - mp_spy_max) / mp_spy_max
    mp_spy_dd_trigger = mp_spy_dd < MP_CONFIG['spy_dd_threshold']  # Today's SPY DD

    # NO SHIFT - returns measured naturally in loop (today's return for yesterday's picks)
    mp_daily_returns = mp_prices.pct_change()

    # Get common dates
    omr_dates = set(spy_df.loc['2017-01-01':'2024-12-31'].index)
    mp_dates = set(mp_prices.loc['2017-01-01':'2024-12-31'].index)
    common_dates = sorted(omr_dates & mp_dates)

    print(f'Processing {len(common_dates)} trading days...')
    print()

    # Track portfolios
    combined_portfolio = INITIAL_CAPITAL
    mp_only_portfolio = INITIAL_CAPITAL
    spy_portfolio = INITIAL_CAPITAL
    total_contributed = INITIAL_CAPITAL

    portfolio_history = []
    current_month = None

    # Track positions for slippage and returns
    prev_mp_positions = []  # Yesterday's stock picks
    prev_mp_exposure = 1.0  # Yesterday's exposure level

    # SLIPPAGE DISABLED FOR THIS RUN
    mp_slippage_pct = 0
    omr_slippage_pct = 0

    total_mp_slippage = 0
    total_omr_slippage = 0

    for date in common_dates:
        # Monthly contribution at start of each new month
        if current_month is None:
            current_month = date.month
        elif date.month != current_month:
            combined_portfolio += MONTHLY_CONTRIBUTION
            mp_only_portfolio += MONTHLY_CONTRIBUTION
            spy_portfolio += MONTHLY_CONTRIBUTION
            total_contributed += MONTHLY_CONTRIBUTION
            current_month = date.month

        # === OMR Return ===
        omr_return = 0.0
        spy_data_omr = omr_data['SPY']
        vix_data_omr = omr_data['^VIX']

        if date in spy_data_omr.index and date in vix_data_omr.index:
            regime, _ = regime_detector.classify_regime(spy_data_omr, vix_data_omr, date)

            if regime not in OMR_CONFIG['skip_regimes']:
                vix_val = float(vix_data_omr[vix_data_omr.index <= date]['close'].iloc[-1])

                if vix_val <= OMR_CONFIG['vix_threshold']:
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
                            'overnight_return': overnight_return,
                            'probability': prob_data['probability']
                        })

                    if day_trades:
                        day_trades.sort(key=lambda x: x['probability'], reverse=True)
                        selected = day_trades[:OMR_CONFIG['max_positions']]
                        # Calculate return with slippage (entry + exit = 2x slippage per trade)
                        omr_return = sum(t['overnight_return'] * OMR_CONFIG['position_size'] for t in selected)
                        # Deduct slippage: 2 trades per position (entry at close, exit at open)
                        slippage_cost = len(selected) * 2 * omr_slippage_pct * OMR_CONFIG['position_size']
                        omr_return -= slippage_cost
                        total_omr_slippage += slippage_cost * combined_portfolio

        # === MP Return ===
        # Step 1: Measure return for YESTERDAY's positions (if any)
        mp_return = 0.0
        if prev_mp_positions and date in mp_daily_returns.index:
            for stock in prev_mp_positions:
                if stock in mp_daily_returns.columns:
                    stock_ret = mp_daily_returns.loc[date, stock]
                    if pd.notna(stock_ret):
                        mp_return += stock_ret * MP_CONFIG['position_size'] * prev_mp_exposure

        # Step 2: Select NEW positions for tomorrow (based on today's momentum)
        current_mp_positions = []
        current_exposure = 1.0

        if date in mp_momentum.index:
            mom_today = mp_momentum.loc[date].dropna()

            if len(mom_today) >= MP_CONFIG['top_n']:
                # Check risk signals using today's data
                risk_active = False
                if date in mp_high_vix.index and mp_high_vix.loc[date]:
                    risk_active = True
                if date in mp_spy_dd_trigger.index and mp_spy_dd_trigger.loc[date]:
                    risk_active = True

                current_exposure = 0.5 if risk_active else 1.0

                top_stocks = mom_today.nlargest(MP_CONFIG['top_n']).index.tolist()
                current_mp_positions = [s for s in top_stocks if s in mp_daily_returns.columns]

        # Calculate slippage for position changes
        prev_set = set(prev_mp_positions)
        curr_set = set(current_mp_positions)
        num_trades = len(curr_set - prev_set) + len(prev_set - curr_set)
        mp_slippage_today = num_trades * mp_slippage_pct * MP_CONFIG['position_size'] * current_exposure
        mp_return -= mp_slippage_today
        total_mp_slippage += mp_slippage_today * mp_only_portfolio

        # Store for next iteration
        prev_mp_positions = current_mp_positions
        prev_mp_exposure = current_exposure

        # === SPY Return ===
        spy_ret = 0.0
        if date in mp_spy.index:
            spy_ret = mp_daily_returns.loc[date, 'SPY'] if pd.notna(mp_daily_returns.loc[date, 'SPY']) else 0

        # === Update Portfolios ===
        combined_return = omr_return + mp_return
        combined_portfolio *= (1 + combined_return)
        mp_only_portfolio *= (1 + mp_return)
        spy_portfolio *= (1 + spy_ret)

        portfolio_history.append({
            'date': date,
            'combined': combined_portfolio,
            'mp_only': mp_only_portfolio,
            'spy': spy_portfolio,
            'contributed': total_contributed
        })

    df = pd.DataFrame(portfolio_history).set_index('date')

    # Yearly summary
    print('='*70)
    print(' YEARLY SUMMARY')
    print('='*70)
    print()
    print(f"{'Year':>6}{'Contributed':>14}{'Combined':>16}{'MP Only':>14}{'SPY':>14}")
    print('-'*66)

    yearly_df = df.resample('Y').last()

    for idx, row in yearly_df.iterrows():
        year = idx.year
        print(f"{year:>6}    ${row['contributed']:>11,.0f}  ${row['combined']:>13,.0f}  ${row['mp_only']:>11,.0f}  ${row['spy']:>11,.0f}")

    # Final summary
    print()
    print('='*70)
    print(' FINAL RESULTS')
    print('='*70)
    print()

    final = df.iloc[-1]

    print(f"Total Contributed:     ${final['contributed']:>14,.0f}")
    print()
    print(f"Combined (OMR+MP):     ${final['combined']:>14,.0f}  ({final['combined']/final['contributed']:.1f}x)")
    print(f"MP Only:               ${final['mp_only']:>14,.0f}  ({final['mp_only']/final['contributed']:.1f}x)")
    print(f"SPY:                   ${final['spy']:>14,.0f}  ({final['spy']/final['contributed']:.1f}x)")
    print()

    combined_gain = final['combined'] - final['contributed']
    mp_gain = final['mp_only'] - final['contributed']
    spy_gain = final['spy'] - final['contributed']

    print(f"Combined Gain:         ${combined_gain:>14,.0f}")
    print(f"MP Only Gain:          ${mp_gain:>14,.0f}")
    print(f"SPY Gain:              ${spy_gain:>14,.0f}")
    print()
    print(f"Combined vs MP:        ${combined_gain - mp_gain:>+14,.0f}")
    print(f"Combined vs SPY:       ${combined_gain - spy_gain:>+14,.0f}")
    print()
    print(f"Total OMR Slippage:    ${total_omr_slippage:>14,.0f}")
    print(f"Total MP Slippage:     ${total_mp_slippage:>14,.0f}")
    print(f"Total Slippage:        ${total_omr_slippage + total_mp_slippage:>14,.0f}")
    print()

    # 2024 monthly income
    final_year = df.loc['2024']
    if len(final_year) > 0:
        monthly_2024 = final_year.resample('M').last()
        combined_monthly = monthly_2024['combined'].diff().mean()
        print(f"Avg Monthly Gain (2024 Combined): ${combined_monthly:>10,.0f}")


if __name__ == '__main__':
    main()
