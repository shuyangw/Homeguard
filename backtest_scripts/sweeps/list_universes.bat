@echo off
REM List Available Universes
REM Shows all predefined universes that can be used with --universe flag

echo.
echo ========================================
echo Available Universes for Backtesting
echo ========================================
echo.
echo MAJOR INDICES:
echo   DOW30          - Dow Jones 30 stocks
echo   NASDAQ100      - Top 50 NASDAQ-100 stocks
echo   SP100          - Top 50 S^&P 100 stocks
echo.
echo POPULAR GROUPS:
echo   FAANG          - META, AAPL, AMZN, NFLX, GOOGL (5 stocks)
echo   MAGNIFICENT7   - AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META (7 stocks)
echo   TECH_GIANTS    - 10 major technology companies
echo.
echo SECTOR UNIVERSES:
echo   SEMICONDUCTORS - 10 semiconductor stocks
echo   ENERGY         - 10 energy stocks
echo   FINANCE        - 10 financial stocks
echo   HEALTHCARE     - 10 healthcare stocks
echo   CONSUMER       - 10 consumer stocks
echo.
echo ========================================
echo Usage Examples:
echo ========================================
echo.
echo Sweep FAANG stocks:
echo   python src\backtest_runner.py --strategy MovingAverageCrossover --universe FAANG --sweep --start 2023-01-01 --end 2024-01-01
echo.
echo Sweep DOW30 in parallel:
echo   python src\backtest_runner.py --strategy BreakoutStrategy --universe DOW30 --sweep --parallel --start 2023-01-01 --end 2024-01-01
echo.
echo Use custom symbols file:
echo   python src\backtest_runner.py --strategy MeanReversion --symbols-file my_watchlist.txt --sweep --start 2023-01-01 --end 2024-01-01
echo.
