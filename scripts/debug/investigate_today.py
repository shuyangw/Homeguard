"""Investigate today's trading activity."""
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

api_key = os.getenv('ALPACA_PAPER_KEY_ID')
secret_key = os.getenv('ALPACA_PAPER_SECRET_KEY')
client = TradingClient(api_key, secret_key, paper=True)

ET = pytz.timezone('America/New_York')
now = datetime.now(ET)
today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

print("=" * 80)
print(f"TRADING INVESTIGATION - {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print("=" * 80)

# Account status
account = client.get_account()
print("\nACCOUNT STATUS")
print("-" * 40)
print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print(f"Day P/L: ${float(account.equity) - float(account.last_equity):,.2f}")

# Current positions
positions = client.get_all_positions()
print(f"\nCURRENT POSITIONS ({len(positions)})")
print("-" * 40)

total_value = 0
total_pnl = 0
for p in sorted(positions, key=lambda x: float(x.market_value), reverse=True):
    value = float(p.market_value)
    pnl = float(p.unrealized_pl)
    pnl_pct = float(p.unrealized_plpc) * 100
    total_value += value
    total_pnl += pnl
    print(f"{p.symbol:6} | {int(p.qty):5} shares | ${value:>10,.2f} | P/L: ${pnl:>8,.2f} ({pnl_pct:>+6.2f}%)")

print(f"\nTotal Position Value: ${total_value:,.2f}")
print(f"Total Unrealized P/L: ${total_pnl:,.2f}")

# Today's orders
print("\n" + "=" * 80)
print("TODAY'S ORDERS")
print("=" * 80)

request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    after=today_start,
    limit=100
)
orders = client.get_orders(filter=request)

filled_orders = [o for o in orders if o.filled_qty and int(o.filled_qty) > 0]
unfilled_orders = [o for o in orders if not o.filled_qty or int(o.filled_qty) == 0]
cancelled_orders = [o for o in orders if 'canceled' in str(o.status).lower() or 'cancelled' in str(o.status).lower()]

print(f"\nFilled Orders: {len(filled_orders)}")
print(f"Unfilled/Pending: {len(unfilled_orders)}")
print(f"Cancelled: {len(cancelled_orders)}")

if filled_orders:
    print("\nFILLED ORDERS:")
    print("-" * 80)
    for o in sorted(filled_orders, key=lambda x: x.filled_at or x.created_at):
        created_et = o.created_at.astimezone(ET)
        side = str(o.side).split('.')[1]
        price = float(o.filled_avg_price) if o.filled_avg_price else 0
        value = int(o.filled_qty) * price
        print(f"{created_et.strftime('%H:%M:%S')} | {o.symbol:6} | {side:4} | {o.filled_qty:>5} @ ${price:>8.2f} = ${value:>10,.2f}")

if cancelled_orders:
    print("\nCANCELLED ORDERS:")
    print("-" * 80)
    for o in cancelled_orders:
        created_et = o.created_at.astimezone(ET)
        side = str(o.side).split('.')[1]
        print(f"{created_et.strftime('%H:%M:%S')} | {o.symbol:6} | {side:4} | qty={o.qty}")

# Check for pending orders (should be 0 after we cancelled)
open_request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
open_orders = client.get_orders(filter=open_request)
print(f"\nCURRENT OPEN ORDERS: {len(open_orders)}")

if open_orders:
    for o in open_orders:
        print(f"  {o.symbol}: {o.side} {o.qty} - {o.status}")

# Recent order history (last 3 days for context)
print("\n" + "=" * 80)
print("RECENT ORDER HISTORY (Last 3 Days)")
print("=" * 80)

three_days_ago = now - timedelta(days=3)
history_request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    after=three_days_ago,
    limit=200
)
history_orders = client.get_orders(filter=history_request)

# Group by date
by_date = {}
for o in history_orders:
    date_str = o.created_at.astimezone(ET).strftime('%Y-%m-%d')
    if date_str not in by_date:
        by_date[date_str] = {'filled': 0, 'cancelled': 0, 'pending': 0}

    if o.filled_qty and int(o.filled_qty) > 0:
        by_date[date_str]['filled'] += 1
    elif 'canceled' in str(o.status).lower():
        by_date[date_str]['cancelled'] += 1
    else:
        by_date[date_str]['pending'] += 1

print(f"\n{'Date':<12} | {'Filled':>8} | {'Cancelled':>10} | {'Pending':>8}")
print("-" * 50)
for date_str in sorted(by_date.keys(), reverse=True):
    stats = by_date[date_str]
    print(f"{date_str:<12} | {stats['filled']:>8} | {stats['cancelled']:>10} | {stats['pending']:>8}")
