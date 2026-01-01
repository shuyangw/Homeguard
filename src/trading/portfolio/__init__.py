"""
Portfolio Tracking Module.

Provides portfolio value tracking with:
- Scheduled updates during equity hours (9:30 AM - 4:00 PM EST)
- On-demand refresh via EC2 commands
- Persistence of value history for analysis
- Integration with CSCM paper trading adapter

Usage:
    from src.trading.portfolio import PortfolioTracker

    # With CSCM adapter
    adapter = CSCMPaperAdapter()
    tracker = PortfolioTracker(adapter)

    # Get current status
    status = tracker.get_status()

    # Force refresh
    value = tracker.update_value(force=True)

    # Check if in equity hours
    if tracker.is_equity_hours():
        print("Market is open")
"""

from src.trading.portfolio.tracker import PortfolioTracker

__all__ = [
    'PortfolioTracker',
]
