"""
Comprehensive Combined OMR + MP (1m-1w) Backtest with Monthly Contributions.

Generates yearly and monthly performance reports with Sharpe, drawdown, and final values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
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
}

INITIAL_CAPITAL = 100000
MONTHLY_CONTRIBUTION = 1000
DATA_DIR = PROJECT_ROOT / 'data' / 'leveraged_etfs'


def calculate_metrics(returns_series, initial_value=INITIAL_CAPITAL, contributions=None):
    """Calculate performance metrics including Sharpe and drawdown."""
    if len(returns_series) == 0:
        return {
            'total_return': 0,
            'sharpe': 0,
            'max_dd': 0,
            'final_value': initial_value
        }

    # Calculate equity curve with contributions
    equity = [initial_value]
    if contributions is None:
        contributions = pd.Series(0, index=returns_series.index)

    for i, (date, ret) in enumerate(returns_series.items()):
        contrib = contributions.loc[date] if date in contributions.index else 0
        new_equity = equity[-1] * (1 + ret) + contrib
        equity.append(new_equity)

    equity = pd.Series(equity[1:], index=returns_series.index)

    # Total return
    total_return = (equity.iloc[-1] - initial_value - contributions.sum()) / (initial_value + contributions.sum())

    # Sharpe ratio (annualized)
    mean_ret = returns_series.mean()
    std_ret = returns_series.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Max drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final_value': equity.iloc[-1]
    }


def main():
    print('='*80)
    print(' COMPREHENSIVE BACKTEST: OMR + MP (1m-1w) WITH MONTHLY CONTRIBUTIONS')
    print('='*80)
    print()
    print(f'Initial Investment: ${INITIAL_CAPITAL:,}')
    print(f'Monthly Contribution: ${MONTHLY_CONTRIBUTION:,}')
    print('Period: 2017-01-01 to 2024-12-31')
    print()
    print('Strategy Configuration:')
    print('  OMR: Bayesian overnight reversion')
    print('    - 15% per position, max 3 positions (45% total exposure)')
    print('    - Entry: 3:50 PM ET, Exit: 9:35 AM ET next day')
    print('    - Trades: Leveraged ETFs (TQQQ, SOXL, etc.)')
    print()
    print('  MP: Momentum Protection with 1m-1w formula')
    print('    - Formula: 21-day return MINUS 5-day return')
    print('    - 6.5% per position, top 10 stocks (65% total exposure)')
    print('    - Daily rebalance at 9:31 AM ET')
    print('    - Risk: VIX > 25 or SPY DD > 5% triggers 50% exposure reduction')
    print('    - Trades: S&P 500 stocks')
    print()
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

    print(f'  Loaded {len(omr_data)-2} OMR symbols')

    # Load MP data
    print('Loading MP data...')
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'
    mp_symbols = pd.read_csv(csv_path)['Symbol'].tolist()

    mp_raw = yf.download(mp_symbols + ['SPY', '^VIX'], start='2016-01-01',
                         end='2024-12-31', progress=False, auto_adjust=True)
    mp_prices = mp_raw['Close'] if isinstance(mp_raw.columns, pd.MultiIndex) else mp_raw
    mp_spy = mp_prices['SPY']
    mp_vix = mp_prices['^VIX']

    print(f'  Downloaded {len(mp_symbols)} MP symbols')

    # Initialize OMR models
    print('Loading Bayesian model...')
    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel()
    bayesian_model.load_model()

    # Pre-compute MP signals (1m-1w momentum)
    print('Computing 1m-1w momentum signals...')
    returns_1m = mp_prices.pct_change(21)
    returns_1w = mp_prices.pct_change(5)
    mp_momentum = returns_1m - returns_1w

    mp_high_vix = mp_vix > MP_CONFIG['vix_threshold']
    mp_spy_max = mp_spy.expanding().max()
    mp_spy_dd = (mp_spy - mp_spy_max) / mp_spy_max
    mp_spy_dd_trigger = mp_spy_dd < MP_CONFIG['spy_dd_threshold']

    mp_daily_returns = mp_prices.pct_change()

    # Get common dates
    omr_dates = set(spy_df.loc['2017-01-01':'2024-12-31'].index)
    mp_dates = set(mp_prices.loc['2017-01-01':'2024-12-31'].index)
    common_dates = sorted(omr_dates & mp_dates)

    print(f'Processing {len(common_dates)} trading days...')
    print()

    # Track portfolios
    combined_portfolio = INITIAL_CAPITAL
    omr_only_portfolio = INITIAL_CAPITAL
    mp_only_portfolio = INITIAL_CAPITAL
    spy_portfolio = INITIAL_CAPITAL
    total_contributed = INITIAL_CAPITAL

    # Track returns for metrics
    daily_records = []
    current_month = None
    contributions_series = pd.Series(0, index=pd.DatetimeIndex(common_dates))

    for date in common_dates:
        # Monthly contribution at start of each new month
        contrib_today = 0
        if current_month is None:
            current_month = date.month
        elif date.month != current_month:
            contrib_today = MONTHLY_CONTRIBUTION
            combined_portfolio += contrib_today
            omr_only_portfolio += contrib_today
            mp_only_portfolio += contrib_today
            spy_portfolio += contrib_today
            total_contributed += contrib_today
            contributions_series.loc[date] = contrib_today
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
                        omr_return = sum(t['overnight_return'] * OMR_CONFIG['position_size'] for t in selected)

        # === MP Return ===
        mp_return = 0.0
        if date in mp_momentum.index:
            mom_today = mp_momentum.loc[date].dropna()

            if len(mom_today) >= MP_CONFIG['top_n']:
                risk_active = False
                if date in mp_high_vix.index and mp_high_vix.loc[date]:
                    risk_active = True
                if date in mp_spy_dd_trigger.index and mp_spy_dd_trigger.loc[date]:
                    risk_active = True

                exposure = 0.5 if risk_active else 1.0

                top_stocks = mom_today.nlargest(MP_CONFIG['top_n']).index.tolist()
                valid_stocks = [s for s in top_stocks if s in mp_daily_returns.columns and date in mp_daily_returns.index]

                for stock in valid_stocks:
                    stock_ret = mp_daily_returns.loc[date, stock]
                    if pd.notna(stock_ret):
                        mp_return += stock_ret * MP_CONFIG['position_size'] * exposure

        # === SPY Return ===
        spy_ret = 0.0
        if date in mp_spy.index:
            spy_ret = mp_daily_returns.loc[date, 'SPY'] if pd.notna(mp_daily_returns.loc[date, 'SPY']) else 0

        # === Update Portfolios ===
        combined_return = omr_return + mp_return

        # Update values BEFORE applying returns (contribution already added)
        combined_portfolio *= (1 + combined_return)
        omr_only_portfolio *= (1 + omr_return)
        mp_only_portfolio *= (1 + mp_return)
        spy_portfolio *= (1 + spy_ret)

        daily_records.append({
            'date': date,
            'combined_return': combined_return,
            'omr_return': omr_return,
            'mp_return': mp_return,
            'spy_return': spy_ret,
            'combined_value': combined_portfolio,
            'omr_value': omr_only_portfolio,
            'mp_value': mp_only_portfolio,
            'spy_value': spy_portfolio,
            'contributed': total_contributed
        })

    df = pd.DataFrame(daily_records).set_index('date')

    # Calculate yearly metrics
    print('='*80)
    print(' YEARLY PERFORMANCE BREAKDOWN')
    print('='*80)
    print()

    yearly_results = []

    for year in range(2017, 2025):
        year_data = df[df.index.year == year]

        if len(year_data) == 0:
            continue

        # Get returns for the year
        combined_rets = year_data['combined_return']
        omr_rets = year_data['omr_return']
        mp_rets = year_data['mp_return']
        spy_rets = year_data['spy_return']

        # Calculate metrics
        combined_sharpe = (combined_rets.mean() / combined_rets.std() * np.sqrt(252)) if combined_rets.std() > 0 else 0
        omr_sharpe = (omr_rets.mean() / omr_rets.std() * np.sqrt(252)) if omr_rets.std() > 0 else 0
        mp_sharpe = (mp_rets.mean() / mp_rets.std() * np.sqrt(252)) if mp_rets.std() > 0 else 0
        spy_sharpe = (spy_rets.mean() / spy_rets.std() * np.sqrt(252)) if spy_rets.std() > 0 else 0

        # Calculate annual return
        combined_ret = (1 + combined_rets).prod() - 1
        omr_ret = (1 + omr_rets).prod() - 1
        mp_ret = (1 + mp_rets).prod() - 1
        spy_ret = (1 + spy_rets).prod() - 1

        # Calculate max drawdown
        combined_equity = (1 + combined_rets).cumprod()
        combined_max_dd = ((combined_equity - combined_equity.expanding().max()) / combined_equity.expanding().max()).min()

        ending_value = year_data['combined_value'].iloc[-1]
        contributed = year_data['contributed'].iloc[-1]

        yearly_results.append({
            'year': year,
            'combined_ret': combined_ret,
            'omr_ret': omr_ret,
            'mp_ret': mp_ret,
            'spy_ret': spy_ret,
            'combined_sharpe': combined_sharpe,
            'omr_sharpe': omr_sharpe,
            'mp_sharpe': mp_sharpe,
            'spy_sharpe': spy_sharpe,
            'max_dd': combined_max_dd,
            'ending_value': ending_value,
            'contributed': contributed
        })

    yearly_df = pd.DataFrame(yearly_results)

    # Print yearly table
    print('COMBINED STRATEGY:')
    print(f"{'Year':>6}  {'Return':>10}  {'Sharpe':>8}  {'Max DD':>10}  {'End Value':>14}  {'Contributed':>14}")
    print('-'*80)

    for _, row in yearly_df.iterrows():
        print(f"{row['year']:>6}  {row['combined_ret']*100:>9.1f}%  {row['combined_sharpe']:>8.2f}  {row['max_dd']*100:>9.1f}%  ${row['ending_value']:>13,.0f}  ${row['contributed']:>13,.0f}")

    print()
    print('COMPONENT STRATEGIES:')
    print(f"{'Year':>6}  {'OMR Ret':>10}  {'OMR Sharpe':>12}  {'MP Ret':>10}  {'MP Sharpe':>12}  {'SPY Ret':>10}  {'SPY Sharpe':>12}")
    print('-'*80)

    for _, row in yearly_df.iterrows():
        print(f"{row['year']:>6}  {row['omr_ret']*100:>9.1f}%  {row['omr_sharpe']:>12.2f}  {row['mp_ret']*100:>9.1f}%  {row['mp_sharpe']:>12.2f}  {row['spy_ret']*100:>9.1f}%  {row['spy_sharpe']:>12.2f}")

    # Monthly breakdown
    print()
    print('='*80)
    print(' MONTHLY PERFORMANCE BREAKDOWN')
    print('='*80)
    print()

    monthly_results = []

    for year in range(2017, 2025):
        year_data = df[df.index.year == year]

        if len(year_data) == 0:
            continue

        print(f"### {year}")
        print(f"{'Month':>6}  {'Return':>10}  {'Sharpe':>8}  {'Max DD':>10}  {'Value':>14}")
        print('-'*60)

        for month in range(1, 13):
            month_data = year_data[year_data.index.month == month]

            if len(month_data) == 0:
                continue

            month_rets = month_data['combined_return']
            month_ret = (1 + month_rets).prod() - 1
            month_sharpe = (month_rets.mean() / month_rets.std() * np.sqrt(21)) if month_rets.std() > 0 else 0

            month_equity = (1 + month_rets).cumprod()
            month_max_dd = ((month_equity - month_equity.expanding().max()) / month_equity.expanding().max()).min()

            month_end_value = month_data['combined_value'].iloc[-1]

            month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%b')

            print(f"{month_name:>6}  {month_ret*100:>9.1f}%  {month_sharpe:>8.2f}  {month_max_dd*100:>9.1f}%  ${month_end_value:>13,.0f}")

            monthly_results.append({
                'year': year,
                'month': month,
                'return': month_ret,
                'sharpe': month_sharpe,
                'max_dd': month_max_dd,
                'value': month_end_value
            })

        print()

    # Monthly statistics
    print('='*80)
    print(' MONTHLY STATISTICS (ALL PERIODS)')
    print('='*80)
    print()

    monthly_df = pd.DataFrame(monthly_results)

    for strategy_name, return_col in [('Combined', 'combined_return'), ('OMR', 'omr_return'),
                                       ('MP (1m-1w)', 'mp_return'), ('SPY', 'spy_return')]:
        monthly_rets = df[return_col].resample('M').apply(lambda x: (1 + x).prod() - 1)

        pos_months = (monthly_rets > 0).sum()
        total_months = len(monthly_rets)
        win_rate = pos_months / total_months * 100 if total_months > 0 else 0

        avg_up = monthly_rets[monthly_rets > 0].mean() * 100 if (monthly_rets > 0).any() else 0
        avg_down = monthly_rets[monthly_rets <= 0].mean() * 100 if (monthly_rets <= 0).any() else 0
        best_month = monthly_rets.max() * 100
        worst_month = monthly_rets.min() * 100

        print(f"{strategy_name}:")
        print(f"  Win Rate:     {win_rate:.1f}% ({pos_months}/{total_months} months)")
        print(f"  Avg Up Month: {avg_up:+.2f}%")
        print(f"  Avg Dn Month: {avg_down:+.2f}%")
        print(f"  Best Month:   {best_month:+.2f}%")
        print(f"  Worst Month:  {worst_month:+.2f}%")
        print()

    # Final summary
    print('='*80)
    print(' FINAL RESULTS (2017-2024)')
    print('='*80)
    print()

    final = df.iloc[-1]

    print(f"Total Contributed:     ${final['contributed']:>14,.0f}")
    print()
    print(f"Combined (OMR+MP):     ${final['combined_value']:>14,.0f}  ({final['combined_value']/final['contributed']:.2f}x)")
    print(f"OMR Only:              ${final['omr_value']:>14,.0f}  ({final['omr_value']/final['contributed']:.2f}x)")
    print(f"MP Only:               ${final['mp_value']:>14,.0f}  ({final['mp_value']/final['contributed']:.2f}x)")
    print(f"SPY:                   ${final['spy_value']:>14,.0f}  ({final['spy_value']/final['contributed']:.2f}x)")
    print()

    combined_gain = final['combined_value'] - final['contributed']
    omr_gain = final['omr_value'] - final['contributed']
    mp_gain = final['mp_value'] - final['contributed']
    spy_gain = final['spy_value'] - final['contributed']

    print(f"Combined Gain:         ${combined_gain:>14,.0f}")
    print(f"OMR Only Gain:         ${omr_gain:>14,.0f}")
    print(f"MP Only Gain:          ${mp_gain:>14,.0f}")
    print(f"SPY Gain:              ${spy_gain:>14,.0f}")
    print()
    print(f"Combined vs OMR:       ${combined_gain - omr_gain:>+14,.0f}")
    print(f"Combined vs MP:        ${combined_gain - mp_gain:>+14,.0f}")
    print(f"Combined vs SPY:       ${combined_gain - spy_gain:>+14,.0f}")
    print()

    # Save report
    report_path = PROJECT_ROOT / 'docs' / 'reports' / '20251203_COMBINED_OMR_MP_1M1W_WITH_CONTRIBUTIONS.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(f"# Backtest Report: Combined OMR + MP (1m-1w) with Monthly Contributions\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Period:** 2017-01-01 to 2024-12-31\n")
        f.write(f"**Initial Capital:** ${INITIAL_CAPITAL:,}\n")
        f.write(f"**Monthly Contribution:** ${MONTHLY_CONTRIBUTION:,}\n\n")

        f.write("## Strategy Configuration\n\n")
        f.write("### OMR (Overnight Mean Reversion)\n")
        f.write("- Uses production Bayesian model at `models/bayesian_reversion_model.pkl`\n")
        f.write("- 3 positions at 15% each (45% total exposure)\n")
        f.write("- Entry: 3:50 PM ET, Exit: 9:35 AM ET next day\n")
        f.write("- Trades leveraged ETFs (TQQQ, SOXL, etc.)\n\n")

        f.write("### MP (Momentum Protection) - 1m-1w Formula\n")
        f.write("- **Formula: 21-day return MINUS 5-day return** (NOT 3m-1m)\n")
        f.write("- 10 positions at 6.5% each (65% total exposure)\n")
        f.write("- Daily rebalance at 9:31 AM ET\n")
        f.write("- Simplified risk: VIX > 25 or SPY DD > 5% triggers 50% exposure reduction\n")
        f.write("- Trades S&P 500 stocks\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Contributed | ${final['contributed']:,.0f} |\n")
        f.write(f"| Final Portfolio Value | ${final['combined_value']:,.0f} |\n")
        f.write(f"| Total Gain | ${combined_gain:,.0f} |\n")
        f.write(f"| Multiple of Contributions | {final['combined_value']/final['contributed']:.2f}x |\n")
        f.write(f"| Combined vs SPY | ${combined_gain - spy_gain:+,.0f} |\n\n")

        f.write("## Yearly Performance\n\n")
        f.write("### Combined Strategy\n\n")
        f.write(f"| Year | Return | Sharpe | Max DD | Ending Value | Contributed |\n")
        f.write(f"|------|--------|--------|--------|--------------|-------------|\n")

        for _, row in yearly_df.iterrows():
            f.write(f"| {row['year']} | {row['combined_ret']*100:+.1f}% | {row['combined_sharpe']:.2f} | {row['max_dd']*100:.1f}% | ${row['ending_value']:,.0f} | ${row['contributed']:,.0f} |\n")

        f.write("\n### Component Strategies\n\n")
        f.write(f"| Year | OMR Return | OMR Sharpe | MP Return | MP Sharpe | SPY Return | SPY Sharpe |\n")
        f.write(f"|------|------------|------------|-----------|-----------|------------|------------|\n")

        for _, row in yearly_df.iterrows():
            f.write(f"| {row['year']} | {row['omr_ret']*100:+.1f}% | {row['omr_sharpe']:.2f} | {row['mp_ret']*100:+.1f}% | {row['mp_sharpe']:.2f} | {row['spy_ret']*100:+.1f}% | {row['spy_sharpe']:.2f} |\n")

        f.write("\n## Monthly Performance\n\n")

        for year in range(2017, 2025):
            year_months = [m for m in monthly_results if m['year'] == year]

            if not year_months:
                continue

            f.write(f"### {year}\n\n")
            f.write(f"| Month | Return | Sharpe | Max DD | Value |\n")
            f.write(f"|-------|--------|--------|--------|-------|\n")

            for m in year_months:
                month_name = pd.Timestamp(year=m['year'], month=m['month'], day=1).strftime('%b')
                f.write(f"| {month_name} | {m['return']*100:+.1f}% | {m['sharpe']:.2f} | {m['max_dd']*100:.1f}% | ${m['value']:,.0f} |\n")

            f.write("\n")

        f.write("## Monthly Statistics\n\n")
        f.write(f"| Metric | Combined | OMR | MP (1m-1w) | SPY |\n")
        f.write(f"|--------|----------|-----|------------|-----|\n")

        # Recalculate for table
        stats_data = {}
        for strategy_name, return_col in [('Combined', 'combined_return'), ('OMR', 'omr_return'),
                                           ('MP', 'mp_return'), ('SPY', 'spy_return')]:
            monthly_rets = df[return_col].resample('M').apply(lambda x: (1 + x).prod() - 1)

            pos_months = (monthly_rets > 0).sum()
            total_months = len(monthly_rets)
            win_rate = pos_months / total_months * 100 if total_months > 0 else 0

            avg_up = monthly_rets[monthly_rets > 0].mean() * 100 if (monthly_rets > 0).any() else 0
            avg_down = monthly_rets[monthly_rets <= 0].mean() * 100 if (monthly_rets <= 0).any() else 0
            best_month = monthly_rets.max() * 100
            worst_month = monthly_rets.min() * 100

            stats_data[strategy_name] = {
                'win_rate': f"{win_rate:.1f}%",
                'avg_up': f"{avg_up:+.2f}%",
                'avg_down': f"{avg_down:+.2f}%",
                'best': f"{best_month:+.2f}%",
                'worst': f"{worst_month:+.2f}%"
            }

        f.write(f"| Win Rate | {stats_data['Combined']['win_rate']} | {stats_data['OMR']['win_rate']} | {stats_data['MP']['win_rate']} | {stats_data['SPY']['win_rate']} |\n")
        f.write(f"| Avg Up Month | {stats_data['Combined']['avg_up']} | {stats_data['OMR']['avg_up']} | {stats_data['MP']['avg_up']} | {stats_data['SPY']['avg_up']} |\n")
        f.write(f"| Avg Down Month | {stats_data['Combined']['avg_down']} | {stats_data['OMR']['avg_down']} | {stats_data['MP']['avg_down']} | {stats_data['SPY']['avg_down']} |\n")
        f.write(f"| Best Month | {stats_data['Combined']['best']} | {stats_data['OMR']['best']} | {stats_data['MP']['best']} | {stats_data['SPY']['best']} |\n")
        f.write(f"| Worst Month | {stats_data['Combined']['worst']} | {stats_data['OMR']['worst']} | {stats_data['MP']['worst']} | {stats_data['SPY']['worst']} |\n")

        f.write("\n## Final Results Comparison\n\n")
        f.write(f"| Strategy | Final Value | Multiple | Total Gain |\n")
        f.write(f"|----------|-------------|----------|------------|\n")
        f.write(f"| Combined (OMR+MP) | ${final['combined_value']:,.0f} | {final['combined_value']/final['contributed']:.2f}x | ${combined_gain:,.0f} |\n")
        f.write(f"| OMR Only | ${final['omr_value']:,.0f} | {final['omr_value']/final['contributed']:.2f}x | ${omr_gain:,.0f} |\n")
        f.write(f"| MP Only | ${final['mp_value']:,.0f} | {final['mp_value']/final['contributed']:.2f}x | ${mp_gain:,.0f} |\n")
        f.write(f"| SPY | ${final['spy_value']:,.0f} | {final['spy_value']/final['contributed']:.2f}x | ${spy_gain:,.0f} |\n")

        f.write("\n---\n")
        f.write("\nGenerated with Claude Code\n")

    print(f"Report saved to: {report_path}")
    print()


if __name__ == '__main__':
    main()
