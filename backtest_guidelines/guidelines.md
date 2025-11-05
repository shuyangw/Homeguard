# Backtesting Pitfalls: A Detailed Guide

Avoiding pitfalls in backtesting is more important than finding a profitable-looking strategy. A backtest that looks "too good to be true" often is, usually because it has fallen into one of the traps below.

---

## 1. Lookahead Bias

This is the most common and dangerous error. It occurs when your simulation uses information that would **not** have been available at the moment of the decision.

* **Definition:** Your strategy, at Time $T$, inadvertently uses data from Time $T+1$ (or later) to make a trade.
* **Why it's Bad:** It makes your strategy appear prescient. You are, in effect, trading on tomorrow's news today.
* **Common Examples:**
    * **Closing Price:** Using the day's `Close` price to decide to buy *at* the `Close` price. You don't know the closing price until the moment the market closes, at which point it's too late to place that trade. The earliest you could *act* on `Close` price information is the `Open` of the *next* bar.
    * **Indicator Peeking:** Calculating a 20-day Simple Moving Average (SMA) and using `data['SMA_20']` on Day 15. The SMA isn't "stable" or finalized until Day 20.
    * **Fundamental Data:** Using a company's "Q1 Earnings" (which ended March 31st) in your model on April 1st. In reality, the company might not *report* those earnings until April 28th. Using the data before it's public is a severe form of lookahead bias.
    * **Data Cleansing:** Seeing a "bad tick" (e.g., a price drop to $0.01) in your data and manually removing it. Your algorithm, running in real-time, would not have had the benefit of this human-curated hindsight.
* **How to Avoid:**
    * Be rigorous with data indexing. When your code is at Day $D$, it should *only* be able to access `data.loc[ :D]`.
    * Use **Point-in-Time (PiT)** databases for fundamental data. These datasets record *when* data was *made available* to the public, not just what period the data *refers to*.
    * Lag all signals. If a signal is generated from the `Close` of Day $D$, ensure your backtest executes the trade at the `Open` of Day $D+1$.

---

## 2. Survivorship Bias

This bias occurs when your dataset excludes entities that have "failed" or been delisted, leaving you with a pool of only the "survivors."

* **Definition:** Running a backtest on a dataset of *current* market components (e.g., the companies *currently* in the S&P 500) over a long historical period.
* **Why it's Bad:** Your results are artificially inflated because your starting universe is pre-filtered for winners. You have excluded all the companies that went bankrupt, were acquired, or were delisted for poor performance—all of which your strategy would have potentially traded and lost money on.
* **Common Examples:**
    * Testing a strategy on the *current* S&P 500 components from 2010 to 2024. This test ignores companies like Enron, Bear Stearns, or Lehman Brothers that were once in the index but are now gone.
    * Using a free stock dataset that doesn't include delisted tickers.
* **How to Avoid:**
    * Use high-quality, professional data that includes delisted securities.
    * When testing on an index, use a **point-in-time constituent list** that shows exactly which companies were in the index on any given day in the past.

---

## 3. Overfitting (Curve-Fitting)

This is the "sin" of data mining. It happens when you create a strategy that is so finely tuned to the *noise* and specific events of your historical data that it has no predictive power on new, unseen data.

* **Definition:** Developing a model with too many parameters or rules that explains the *past* perfectly but fails to generalize to the *future*.
* **Why it's Bad:** You've built a "one-trick pony" that has memorized history rather than learned a robust, repeatable market dynamic. It will almost certainly fail in live trading.
* **Common Examples:**
    * A strategy with many "magic numbers" (e.g., "Buy when 7-day RSI < 18 AND 42-day SMA is rising AND it's a Tuesday").
    * Testing thousands of parameter combinations and picking the *single* best one (e.g., finding that a `(12, 26)` MACD performs poorly, but a `(13, 29)` MACD is a goldmine).
    * Your strategy's performance is a "brittle" (highly sensitive) function of its parameters.
* **How to Avoid:**
    * **Keep it Simple (Occam's Razor):** A strategy with 2-3 parameters is more likely to be robust than one with 10.
    * **Strict Data Splitting:** Use a non-negotiable **Out-of-Sample (OOS)** dataset. Develop and optimize your strategy *only* on the In-Sample (training) data. Then, run the *final, locked-in* strategy *once* on the OOS (test) data. If the performance collapses, your model is overfit.
    * **Walk-Forward Optimization:** This is a more advanced OOS technique. You optimize on Years 1-3, then "trade" on Year 4. Then, you roll the window forward: optimize on Years 2-4, "trade" on Year 5. This simulates how a strategy would be re-calibrated in a live environment.
    * **Parameter Stability:** Test the "neighborhood" around your chosen parameters. If `SMA_Period = 50` works, `48` and `52` should also be profitable, even if slightly less so. If they are catastrophic failures, your `50` is likely an overfit fluke.

---

## 4. Unrealistic Cost & Execution Modeling

This pitfall involves ignoring the "frictions" of real-world trading, which can turn a profitable backtest into a money-losing-live strategy.

* **Definition:** Assuming you can trade for free, execute at the exact price you see in your data, and trade any size without issue.
* **Why it's Bad:** Costs are a *guaranteed* loss on every trade. For high-frequency strategies, these costs are often the *only* thing that matters.
* **Common Examples:**
    * **Ignoring Commissions:** Failing to subtract broker fees for every buy and sell.
    * **Ignoring Slippage:** This is critical. You assume you can buy at the `last_price` of $50.00. But in reality, to execute a *market buy order*, you have to cross the spread and buy at the `ask` price of $50.02. This $0.02 is slippage.
    * **Ignoring Market Impact:** Assuming you can buy $10M worth of a small-cap stock without the price moving. In reality, your own large order will consume all available liquidity and drive the price up *against* you.
    * **Ignoring Liquidity:** Your model decides to buy 10,000 shares, but the `ask` side only has 500 shares available at that price. Your backtest must realistically model partial fills or filling at much worse prices.
* **How to Avoid:**
    * At a minimum, model **commissions** (e.g., $0.005 per share) and a **slippage** estimate (e.g., 1-2 basis points on the trade value, or assuming you *always* cross the bid-ask spread).
    * For more realism, use Level 2 (order book) data to model your fills against available liquidity.
    * Be *more* conservative with your cost assumptions, not less.

---

## 5. Ignoring Market Regimes

This is a strategic-level failure where a strategy is tested or developed on a period that is not representative of all market conditions.

* **Definition:** Building a strategy that works perfectly in one "regime" (e.g., a low-volatility bull market) but failing to test how it performs in others (e.g., a high-volatility crash or a sideways "chop").
* **Why it's Bad:** The market environment is not static. A "trend-following" strategy will look brilliant from 2016-2021 but may have catastrophic drawdowns in a sideways market. A "mean-reversion" strategy will do the opposite.
* **Common Examples:**
    * Developing a "buy the dip" strategy using data *only* from the 2011-2021 bull market. This strategy would be destroyed in 2008 or 2022.
    * Optimizing a VIX-based strategy on 2018 (high-vol) data and assuming it will work the same in 2017 (record-low-vol).
* **How to Avoid:**
    * Your backtest **must** include multiple regimes: a bull market, a bear market/crash (e.g., 2008, 2020, 2022), and a sideways or "choppy" market.
    * Analyze your strategy's performance *conditional* on market conditions (e.g., "How does it do when VIX > 30?" or "How does it do when the 200-day trend is down?").
    * This is why having *at least* 10-15+ years of data is so important.

---

## 6. Unrealistic Position Sizing

This is a critical pitfall that invalidates backtest results by using position sizes that would never be used in live trading.

* **Definition:** Allocating an unrealistic percentage of capital to each trade (commonly 99% or 100%), which inflates performance metrics and makes diversification impossible.
* **Why it's Bad:** Using 99% of capital per trade means:
  - **No diversification**: You can only hold one position at a time, concentrating all risk
  - **Inflated returns**: Single 50% gains multiply your capital by 1.5×, which isn't achievable with realistic sizing
  - **Catastrophic drawdowns**: Single 10% loss = 9.9% portfolio loss
  - **Not actionable**: No professional trader would ever use 99% per trade in live trading
  - **Violates risk management**: Professional traders typically use 1-10% per trade
* **Common Examples:**
    * **All-in per trade**: `shares = (cash * 0.99) / price` - uses 99% of capital per trade
    * **No position limits**: Allowing positions to exceed 50% of portfolio
    * **Ignoring correlations**: Taking 5× 20% positions in highly correlated stocks (effectively 100% exposure)
    * **No stop losses**: Letting losers run without maximum loss limits
    * **Fixed dollar amounts on small accounts**: Using $10k per trade on a $20k account (50% per trade)
* **Real-World Impact:**
    ```
    Example:
    Backtest with 99% position sizing:
    - Capital: $100,000
    - Trade 1: Buy AAPL with $99,000 (660 shares @ $150)
    - AAPL rises 20% → Portfolio = $118,800 (+18.8%)

    Same strategy with realistic 10% position sizing:
    - Capital: $100,000
    - Trade 1: Buy AAPL with $10,000 (66 shares @ $150)
    - AAPL rises 20% → Portfolio = $102,000 (+2.0%)

    Backtest Sharpe with 99% sizing: 3.5 (impressive!)
    Real Sharpe with 10% sizing: 0.8 (mediocre)

    Result: Strategy looks amazing in backtest but underperforms in live trading.
    ```
* **How to Avoid:**
    * **Use Fixed Percentage Sizing:** Allocate 5-10% of capital per trade (max 20% for aggressive strategies)
    ```python
    from backtesting.utils.risk_config import RiskConfig

    # Conservative: 5% per trade
    config = RiskConfig.conservative()

    # Moderate: 10% per trade (DEFAULT)
    config = RiskConfig.moderate()

    # Aggressive: 20% per trade
    config = RiskConfig.aggressive()

    engine = BacktestEngine(initial_capital=100000, risk_config=config)
    ```
    * **Use Volatility-Based Sizing:** Risk a fixed percentage (1-2%) based on volatility
    ```python
    from backtesting.utils.position_sizer import VolatilityBasedSizer

    sizer = VolatilityBasedSizer(
        risk_pct=0.01,        # Risk 1% per trade
        atr_multiplier=2.0     # 2× ATR stop distance
    )
    shares = sizer.calculate_shares(portfolio_value, price, price_data)
    ```
    * **Use Kelly Criterion (Half Kelly):** Mathematically optimal sizing based on edge
    ```python
    from backtesting.utils.position_sizer import KellyCriterionSizer

    sizer = KellyCriterionSizer(
        win_rate=0.55,
        avg_win=500,
        avg_loss=300,
        kelly_fraction=0.5  # Half Kelly for safety (never use Full Kelly!)
    )
    shares = sizer.calculate_shares(portfolio_value, price)
    ```
    * **Set Position Limits:** Never exceed maximum position size
    ```python
    config = RiskConfig(
        position_size_pct=0.10,          # 10% per trade
        max_single_position_pct=0.25,    # Never exceed 25% per position
        max_positions=10                  # Max 10 concurrent positions
    )
    ```
    * **Apply Stop Losses:** Limit maximum loss per trade
    ```python
    config = RiskConfig(
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,  # Exit when down 2%
        stop_loss_type='atr'  # Or 'percentage', 'time', 'profit_target'
    )
    ```
* **Position Sizing Methods:**
    1. **Fixed Percentage**: 10% per trade (simplest, most common)
    2. **Fixed Dollar**: $10,000 per trade (for small accounts, doesn't scale)
    3. **Volatility-Based (ATR)**: Equalize risk across trades (professional approach)
    4. **Kelly Criterion**: Optimal mathematical sizing (requires accurate statistics)
    5. **Risk Parity**: Equal risk contribution across assets (multi-asset portfolios)

    See `docs/POSITION_SIZING_METHODS.md` for detailed formulas and examples.
* **Red Flags:**
    - Backtest Sharpe ratio > 3.0 with high single-trade impact
    - Single trades moving portfolio by >5%
    - Strategy can't be run on multiple symbols simultaneously
    - Returns drop dramatically when reducing position size
    - Drawdowns are tiny (<5%) despite volatile underlying assets

---

### Quick-Reference Summary

| Pitfall | Why It's Dangerous | How to Avoid |
| :--- | :--- | :--- |
| **Lookahead Bias** | Trades on "future" data. Looks prescient. | Lag all signals by 1 bar. Use Point-in-Time data. |
| **Survivorship Bias** | Tests only on "winners," inflating returns. | Use datasets that include delisted/failed companies. |
| **Overfitting** | Memorizes past noise, not a real pattern. | Use strict In-Sample/Out-of-Sample splits. Keep it simple. |
| **Unrealistic Costs** | Ignores commissions/slippage. Turns gross profits into net losses. | Model *all* costs: commissions, slippage, and spread. |
| **Ignoring Regimes** | Strategy is a "one-trick pony" for a specific market. | Test across 10+ years, ensuring you cover crashes, bubbles, & sideways markets. |
| **Unrealistic Position Sizing** | Uses 99% per trade. Inflates returns and Sharpe. Not actionable. | Use 5-10% per trade with `RiskConfig.moderate()`. Apply stop losses. |