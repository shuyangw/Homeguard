# Alpaca Paper Trading Dashboard Guide

This guide explains how to view and monitor your paper trading activity on the Alpaca website.

## Accessing the Paper Trading Dashboard

### 1. Log In to Alpaca

1. Go to [https://app.alpaca.markets/](https://app.alpaca.markets/)
2. Log in with your credentials
3. **IMPORTANT**: Make sure you're viewing the **Paper Trading** account (not Live Trading)
   - Look for a toggle or dropdown in the top right corner
   - It should say "Paper" or "Paper Trading"

### 2. Key Pages to Monitor

#### Portfolio Dashboard
**URL**: [https://app.alpaca.markets/paper/dashboard](https://app.alpaca.markets/paper/dashboard)

**What You'll See**:
- **Portfolio Value**: Total value of your paper trading account
- **Buying Power**: Available cash for trading
- **Today's P&L**: Profit/loss for the current trading day
- **Total P&L**: All-time profit/loss
- **Positions Chart**: Visual representation of your holdings

#### Positions Page
**URL**: [https://app.alpaca.markets/paper/positions](https://app.alpaca.markets/paper/positions)

**What You'll See**:
- **Open Positions**: All currently held stocks/ETFs
  - Symbol, quantity, entry price, current price
  - Unrealized P&L (gain/loss while holding)
  - Position value as % of portfolio
- **Closed Positions**: Historical trades
  - Entry/exit prices and dates
  - Realized P&L (actual profit/loss from closed trades)

#### Orders Page
**URL**: [https://app.alpaca.markets/paper/orders](https://app.alpaca.markets/paper/orders)

**What You'll See**:
- **Pending Orders**: Orders waiting to be filled
- **Filled Orders**: Completed buy/sell orders
- **Cancelled Orders**: Orders that were cancelled
- **Failed Orders**: Orders that failed to execute

For each order:
- Order ID, symbol, side (buy/sell)
- Order type (market, limit, etc.)
- Quantity, price, timestamp
- Status (filled, pending, cancelled, failed)

#### Account Page
**URL**: [https://app.alpaca.markets/paper/account](https://app.alpaca.markets/paper/account)

**What You'll See**:
- Detailed account information
- API keys (if you need to check them)
- Trading settings and configuration

### 3. Monitoring Live Trading Activity

When you run the paper trading adapters, you can monitor activity in real-time:

1. **Start your strategy**:
   ```bash
   python scripts/trading/demo_ma_paper_trading.py --continuous
   ```

2. **Open the Alpaca dashboard** in your browser

3. **Watch for updates**:
   - Refresh the Orders page to see new orders appear
   - Check the Positions page to see when orders are filled
   - Monitor the Portfolio Dashboard for P&L updates

---

### 3.5. Monitoring Cloud-Deployed Bot

If you deployed your trading bot to AWS EC2, use the SSH management scripts to monitor bot activity remotely:

#### Quick Status Check

**Windows**:
```bash
# From repository root
scripts\ec2\check_bot.bat
```

**Linux/Mac**:
```bash
# From repository root
scripts/ec2/check_bot.sh
```

**Shows**:
- Systemd service status (running/stopped)
- Process ID and memory usage
- Last 10 log lines with recent activity

#### Live Log Monitoring

**Windows**:
```bash
scripts\ec2\view_logs.bat
```

**Linux/Mac**:
```bash
scripts/ec2/view_logs.sh
```

**Shows**:
- Real-time bot logs streaming to your terminal
- Market status, checks, signals, and orders
- Press Ctrl+C to stop viewing

#### Daily Health Check

**Windows**:
```bash
scripts\ec2\daily_health_check.bat
```

**Linux/Mac**:
```bash
scripts/ec2/daily_health_check.sh
```

**Performs 6-point validation**:
1. EC2 instance state (running/stopped)
2. Bot service status (active/failed)
3. Recent errors count (last hour)
4. Resource usage (memory/CPU)
5. Last activity (recent logs)
6. Current market status (OPEN/CLOSED)

#### Combining Alpaca Dashboard + SSH Scripts

**Best monitoring workflow**:

1. **Morning (before market open)**:
   - Run `daily_health_check.bat/.sh` to verify bot is ready
   - Check Alpaca Portfolio Dashboard for starting balance

2. **During market hours**:
   - Use `view_logs.bat/.sh` to see bot activity in real-time
   - Check Alpaca Orders page periodically to see trades
   - Monitor Alpaca Positions page for open positions

3. **After market close**:
   - Check Alpaca Dashboard for daily P&L
   - Review bot logs for any errors or issues
   - Verify instance stopped at 4:30 PM ET (if using automated scheduling)

#### Comprehensive Monitoring Guide

For complete monitoring documentation, see:
- **[SSH Scripts README](../../scripts/ec2/SSH_SCRIPTS_README.md)** - All management scripts reference
- **[Health Check Cheatsheet](../HEALTH_CHECK_CHEATSHEET.md)** - Comprehensive monitoring guide
- **[Infrastructure Overview](../INFRASTRUCTURE_OVERVIEW.md)** - AWS architecture and operations

#### Troubleshooting Cloud Deployment

If bot is not trading:
```bash
# 1. Check if bot is running
scripts\ec2\check_bot.bat  # or .sh

# 2. View recent errors
# SSH to instance and check errors (use EC2_IP from .env):
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>
sudo journalctl -u homeguard-trading -p err -n 20

# 3. Restart if needed
scripts\ec2\restart_bot.bat  # or .sh
```

---

### 4. Testing Off-Hours

Since markets are closed, you can test the integration using the off-hours test script:

```bash
# Run the mock test (doesn't need Alpaca)
python scripts/trading/test_paper_trading_off_hours.py mock

# Run MA Crossover test with Alpaca (bypasses market hours)
python scripts/trading/test_paper_trading_off_hours.py ma

# Run OMR test with Alpaca (bypasses market hours)
python scripts/trading/test_paper_trading_off_hours.py omr

# Run all tests
python scripts/trading/test_paper_trading_off_hours.py
```

**Note**: Off-hours tests will bypass the market hours check, allowing you to test the integration even when markets are closed.

### 5. What to Look For

#### Successful Order Placement
When the strategy generates a signal and places an order:

1. **In your console**:
   ```
   Executing BUY 10 shares of AAPL @ $150.25
   [+] Order placed: abc123-def456-...
   ```

2. **On Alpaca Orders page**:
   - New order appears with status "new" or "pending_new"
   - Order transitions to "filled" when executed
   - You'll see the fill price and timestamp

3. **On Alpaca Positions page**:
   - New position appears after order fills
   - Shows quantity, entry price, current value
   - Updates in real-time with market prices

#### Position Monitoring
Once you have open positions:

1. **Portfolio Dashboard** shows:
   - Total portfolio value
   - Today's P&L (how much you're up/down today)
   - Position allocation pie chart

2. **Positions Page** shows each position:
   - Unrealized P&L (current profit/loss)
   - % gain/loss since entry
   - Position size as % of portfolio

#### Closing Positions
When the strategy exits a position:

1. **In your console**:
   ```
   Closing AAPL: 10 shares @ $150.25 -> $155.75 (P&L: +$55.00, +3.66%)
   [+] Close order placed: xyz789-abc123-...
   ```

2. **On Alpaca**:
   - Sell order appears on Orders page
   - Position removed from Open Positions
   - Position moves to Closed Positions with realized P&L

### 6. Common Issues

#### Not Seeing Orders
**Problem**: Ran the strategy but no orders appear on Alpaca

**Possible Causes**:
1. Market hours check prevented execution
   - Solution: Use off-hours test script to bypass
2. No signals were generated
   - Solution: Check console output for "Generated X signals"
3. Signals were filtered out
   - Solution: Check for "Skipping: Already have position" or "Max positions reached"
4. Using wrong API keys
   - Solution: Verify `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`
5. Not in paper trading mode
   - Solution: Check that broker initialized with `mode='paper'`

#### Orders Stuck as "Pending"
**Problem**: Orders show as "pending" but never fill

**Possible Causes**:
1. Market is closed (for paper trading, orders should fill immediately during market hours)
2. Using limit orders on illiquid symbols
   - Solution: Use market orders for testing
3. Symbol is halted or not tradeable
   - Solution: Test with liquid symbols like AAPL, MSFT, SPY

#### Wrong Account (Live vs Paper)
**Problem**: Can't find your test trades

**Solution**:
- Make sure you're viewing the **Paper Trading** account
- Look for the account switcher (usually top right)
- Paper trades won't show in Live account and vice versa

### 7. Best Practices for Testing

1. **Start Small**:
   - Use small position sizes (e.g., `position_size=0.05` for 5%)
   - Limit max positions (e.g., `max_positions=2`)

2. **Test One Symbol First**:
   - Start with a single well-known stock (AAPL, MSFT)
   - Verify orders appear and fill correctly
   - Then expand to more symbols

3. **Monitor Console and Dashboard**:
   - Keep console output visible for immediate feedback
   - Keep Alpaca dashboard open to verify orders
   - Look for any error messages in console

4. **Reset if Needed**:
   - Alpaca paper trading accounts reset periodically
   - You can manually close all positions via the dashboard
   - Or reset API keys to start fresh

### 8. Example: End-to-End Test

Here's a complete example of testing the MA Crossover strategy:

```bash
# 1. Run the off-hours test to verify integration
python scripts/trading/test_paper_trading_off_hours.py ma

# 2. Check console output for:
# - "Connected to Alpaca Paper Trading"
# - "Running strategy..."
# - Order placement messages

# 3. Open Alpaca dashboard
# - Go to https://app.alpaca.markets/paper/orders
# - Look for orders with timestamp matching test run

# 4. Verify order details match console output
# - Symbol, quantity, side (buy/sell)
# - Order status (filled)

# 5. Check positions page
# - Open positions should show newly filled orders
# - Entry price should match fill price
```

## Summary

- **Dashboard**: Monitor portfolio value and P&L
- **Positions**: View open and closed positions
- **Orders**: Track order status and history
- **Account**: Check settings and API configuration
- **Off-Hours Testing**: Use test script to bypass market hours check

For any issues, check:
1. Console output for errors
2. API keys are correct in `.env`
3. You're viewing the Paper Trading account
4. Market hours (or use off-hours test script)
