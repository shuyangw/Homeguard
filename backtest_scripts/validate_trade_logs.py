"""
Validate overnight mean reversion trade logs for strategy compliance.

Checks:
1. No trades in BEAR regime
2. VIX always < 35
3. Probability >= 58%
4. Expected return >= 0.2%
5. Position size always 10%
6. Stop-loss triggers at -3%
7. Sample size >= 15 (need to check Bayesian model)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils.logger import logger

def validate_trade_log(csv_path, version_name):
    """Validate a single trade log CSV."""

    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATING: {version_name}")
    logger.info(f"{'='*80}\n")

    # Load data
    df = pd.read_csv(csv_path)
    total_trades = len(df)

    logger.info(f"Total trades loaded: {total_trades}\n")

    # Track violations
    violations = []

    # ===================================================================
    # Rule 1: No trades in BEAR regime
    # ===================================================================
    bear_trades = df[df['regime'] == 'BEAR']
    if len(bear_trades) > 0:
        violations.append(f"VIOLATION: {len(bear_trades)} trades in BEAR regime")
        logger.error(f"[FAIL] Found {len(bear_trades)} trades in BEAR regime!")
        logger.error(f"  Dates: {bear_trades['date'].tolist()}")
    else:
        logger.info(f"[PASS] No trades in BEAR regime")

    # ===================================================================
    # Rule 2: VIX always < 35
    # ===================================================================
    high_vix_trades = df[df['vix'] >= 35]
    if len(high_vix_trades) > 0:
        violations.append(f"VIOLATION: {len(high_vix_trades)} trades with VIX >= 35")
        logger.error(f"[FAIL] Found {len(high_vix_trades)} trades with VIX >= 35!")
        logger.error(f"  Max VIX: {df['vix'].max():.2f}")
        logger.error(f"  Dates: {high_vix_trades['date'].tolist()}")
    else:
        logger.info(f"[PASS] All trades have VIX < 35 (max: {df['vix'].max():.2f})")

    # ===================================================================
    # Rule 3: Probability >= 58%
    # ===================================================================
    low_prob_trades = df[df['probability'] < 0.58]
    if len(low_prob_trades) > 0:
        violations.append(f"VIOLATION: {len(low_prob_trades)} trades with probability < 58%")
        logger.error(f"[FAIL] Found {len(low_prob_trades)} trades with probability < 58%!")
        logger.error(f"  Min probability: {df['probability'].min():.2%}")
        logger.error(f"  Examples:")
        for _, row in low_prob_trades.head(5).iterrows():
            logger.error(f"    {row['date']}, {row['symbol']}: {row['probability']:.2%}")
    else:
        logger.info(f"[PASS] All trades have probability >= 58% (min: {df['probability'].min():.2%})")

    # ===================================================================
    # Rule 4: Expected return >= 0.2%
    # ===================================================================
    low_exp_trades = df[df['expected_return'] < 0.002]
    if len(low_exp_trades) > 0:
        violations.append(f"VIOLATION: {len(low_exp_trades)} trades with expected_return < 0.2%")
        logger.error(f"[FAIL] Found {len(low_exp_trades)} trades with expected_return < 0.2%!")
        logger.error(f"  Min expected return: {df['expected_return'].min():.4%}")
        logger.error(f"  Examples:")
        for _, row in low_exp_trades.head(5).iterrows():
            logger.error(f"    {row['date']}, {row['symbol']}: {row['expected_return']:.4%}")
    else:
        logger.info(f"[PASS] All trades have expected_return >= 0.2% (min: {df['expected_return'].min():.4%})")

    # ===================================================================
    # Rule 5: Position size always 10%
    # ===================================================================
    wrong_size_trades = df[df['position_size'] != 0.1]
    if len(wrong_size_trades) > 0:
        violations.append(f"VIOLATION: {len(wrong_size_trades)} trades with position_size != 10%")
        logger.error(f"[FAIL] Found {len(wrong_size_trades)} trades with wrong position size!")
        logger.error(f"  Position sizes found: {df['position_size'].unique()}")
    else:
        logger.info(f"[PASS] All trades use 10% position size")

    # ===================================================================
    # Rule 6: Stop-loss triggers at exactly -3%
    # ===================================================================
    stopped_trades = df[df['stopped_out'] == True]
    if len(stopped_trades) > 0:
        # Check that all stopped trades have actual_return = -0.03
        incorrect_stops = stopped_trades[stopped_trades['actual_return'] != -0.03]
        if len(incorrect_stops) > 0:
            violations.append(f"VIOLATION: {len(incorrect_stops)} stopped trades don't have -3% return")
            logger.error(f"[FAIL] Found {len(incorrect_stops)} stopped trades with incorrect returns!")
            for _, row in incorrect_stops.iterrows():
                logger.error(f"    {row['date']}, {row['symbol']}: {row['actual_return']:.2%}")
        else:
            logger.info(f"[PASS] All {len(stopped_trades)} stopped trades capped at -3%")
            logger.info(f"  Stop-out rate: {len(stopped_trades)/total_trades:.1%}")
    else:
        logger.info(f"[INFO] No trades hit stop-loss in this dataset")

    # ===================================================================
    # Rule 7: Max 3 concurrent positions (check by date)
    # ===================================================================
    trades_per_day = df.groupby('date').size()
    max_concurrent = trades_per_day.max()
    days_exceeding_3 = trades_per_day[trades_per_day > 3]

    if len(days_exceeding_3) > 0:
        violations.append(f"VIOLATION: {len(days_exceeding_3)} days with > 3 positions")
        logger.error(f"[FAIL] Found {len(days_exceeding_3)} days with > 3 concurrent positions!")
        logger.error(f"  Max concurrent: {max_concurrent}")
        logger.error(f"  Dates:")
        for date, count in days_exceeding_3.head(10).items():
            logger.error(f"    {date}: {count} positions")
    else:
        logger.info(f"[PASS] No more than 3 concurrent positions (max: {max_concurrent})")

    # ===================================================================
    # Additional Checks: Data Quality
    # ===================================================================
    logger.info("\n--- Additional Quality Checks ---\n")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"[WARN] Found missing values:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"  {col}: {count} missing")
    else:
        logger.info(f"[PASS] No missing values")

    # Check regime distribution
    logger.info(f"\nRegime Distribution:")
    regime_counts = df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / total_trades * 100
        logger.info(f"  {regime}: {count} ({pct:.1f}%)")

    # Check symbol distribution
    logger.info(f"\nTop 10 Symbols:")
    symbol_counts = df['symbol'].value_counts().head(10)
    for symbol, count in symbol_counts.items():
        pct = count / total_trades * 100
        logger.info(f"  {symbol}: {count} ({pct:.1f}%)")

    # Check profitability
    profitable = df['profitable'].sum()
    win_rate = profitable / total_trades
    logger.info(f"\nPerformance Summary:")
    logger.info(f"  Winning trades: {profitable}")
    logger.info(f"  Losing trades: {total_trades - profitable}")
    logger.info(f"  Win rate: {win_rate:.2%}")
    logger.info(f"  Avg return per trade: {df['actual_return'].mean():.4%}")
    logger.info(f"  Total return: {df['portfolio_return'].sum():.2%}")

    # ===================================================================
    # Summary
    # ===================================================================
    logger.info(f"\n{'='*80}")
    if len(violations) == 0:
        logger.info("[SUCCESS] All strategy rules validated successfully!")
    else:
        logger.error(f"[FAILED] Found {len(violations)} violations:")
        for v in violations:
            logger.error(f"  - {v}")
    logger.info(f"{'='*80}\n")

    return len(violations) == 0


def main():
    """Validate all available trade logs."""

    reports_dir = Path('reports')

    # V2 trades
    v2_file = reports_dir / 'overnight_validation_trades_v2.csv'
    if v2_file.exists():
        validate_trade_log(v2_file, "V2 Risk-Managed")
    else:
        logger.warning(f"V2 trade log not found: {v2_file}")

    # V3 trades - check each configuration
    v3_configs = [
        ('overnight_v3_all_23_etfs_trades.csv', 'V3 - All 23 ETFs'),
        ('overnight_v3_top_5_(v2_baseline)_trades.csv', 'V3 - Top 5 (V2 Baseline)'),
        ('overnight_v3_3x_long_only_trades.csv', 'V3 - 3x Long Only'),
        ('overnight_v3_3x_short_only_trades.csv', 'V3 - 3x Short Only'),
        ('overnight_v3_2x_leveraged_trades.csv', 'V3 - 2x Leveraged'),
        ('overnight_v3_tech_sector_trades.csv', 'V3 - Tech Sector'),
        ('overnight_v3_broad_market_trades.csv', 'V3 - Broad Market'),
    ]

    for filename, version_name in v3_configs:
        v3_file = reports_dir / filename
        if v3_file.exists():
            validate_trade_log(v3_file, version_name)
        else:
            logger.warning(f"V3 trade log not found: {v3_file}")


if __name__ == '__main__':
    main()
