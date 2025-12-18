"""Check what signals RAMP generated today."""
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
print("SIGNAL ANALYSIS")
print("=" * 80)

# Get all orders from today
request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    after=today_start,
    limit=100
)
orders = client.get_orders(filter=request)

# Analyze orders
sells = [o for o in orders if 'sell' in str(o.side).lower()]
buys = [o for o in orders if 'buy' in str(o.side).lower()]

print(f"\nSells attempted: {len(sells)}")
for o in sells:
    status = str(o.status).split('.')[1]
    filled = int(o.filled_qty) if o.filled_qty else 0
    print(f"  {o.symbol}: {o.qty} -> filled {filled} ({status})")

print(f"\nBuys attempted: {len(buys)}")
for o in buys:
    status = str(o.status).split('.')[1]
    filled = int(o.filled_qty) if o.filled_qty else 0
    print(f"  {o.symbol}: {o.qty} -> filled {filled} ({status})")

# Check for failed orders
failed = [o for o in orders if 'rejected' in str(o.status).lower() or 'canceled' in str(o.status).lower()]
if failed:
    print(f"\nFailed/Cancelled orders: {len(failed)}")
    for o in failed:
        print(f"  {o.symbol}: {o.side} {o.qty} - {o.status}")

# Current positions breakdown
print("\n" + "=" * 80)
print("POSITION BREAKDOWN")
print("=" * 80)

positions = client.get_all_positions()
position_symbols = {p.symbol for p in positions}
sold_symbols = {o.symbol for o in sells}
bought_symbols = {o.symbol for o in buys}

held_symbols = position_symbols - bought_symbols
print(f"\nPositions held (not traded today): {held_symbols}")
print(f"Positions bought today: {bought_symbols}")
print(f"Positions sold today: {sold_symbols}")

# What SHOULD have happened
print("\n" + "=" * 80)
print("EXPECTED vs ACTUAL")
print("=" * 80)

# In WEAK_BULL with top_n=10:
# Should have 10 positions total
# We have 4, so missing 6

print(f"\nIf WEAK_BULL (top_n=10):")
print(f"  Expected positions: 10")
print(f"  Actual positions: {len(positions)}")
print(f"  Missing: {10 - len(positions)} positions")
print(f"  Cash sitting idle: ~{10 - len(positions)} x $10k = ${(10 - len(positions)) * 10000:,}")

# Check Dec 16 (yesterday) for comparison
print("\n" + "=" * 80)
print("YESTERDAY (Dec 16) COMPARISON")
print("=" * 80)

yesterday_start = today_start - timedelta(days=1)
yesterday_request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    after=yesterday_start,
    until=today_start,
    limit=100
)
yesterday_orders = client.get_orders(filter=yesterday_request)

yesterday_sells = [o for o in yesterday_orders if 'sell' in str(o.side).lower()]
yesterday_buys = [o for o in yesterday_orders if 'buy' in str(o.side).lower()]

print(f"\nYesterday sells: {len(yesterday_sells)}")
print(f"Yesterday buys: {len(yesterday_buys)}")

for o in yesterday_buys:
    status = str(o.status).split('.')[1]
    filled = int(o.filled_qty) if o.filled_qty else 0
    print(f"  {o.symbol}: {o.qty} -> filled {filled} ({status})")
