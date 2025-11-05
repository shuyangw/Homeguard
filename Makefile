.PHONY: help test test-verbose test-engine test-strategies test-pnl backtest-quick backtest-all ingest clean lint

help:
	@echo "Homeguard Makefile - Common Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test              Run all unit tests"
	@echo "  make test-verbose      Run tests with verbose output"
	@echo "  make test-engine       Run backtest engine tests only"
	@echo "  make test-strategies   Run strategy tests only"
	@echo "  make test-pnl          Run P&L calculation tests only"
	@echo ""
	@echo "Backtesting:"
	@echo "  make backtest-quick    Run quick test backtest"
	@echo "  make backtest-all      Run all basic backtests"
	@echo ""
	@echo "Data Ingestion:"
	@echo "  make ingest            Run data ingestion"
	@echo ""
	@echo "Utilities:"
	@echo "  make lint              Run Python linter"
	@echo "  make clean             Clean Python cache files"

test:
	python -m pytest tests/

test-verbose:
	python -m pytest tests/ -v

test-engine:
	python -m pytest tests/test_backtest_engine.py -v

test-strategies:
	python -m pytest tests/test_strategies.py -v

test-pnl:
	python -m pytest tests/test_pnl_calculations.py -v

backtest-quick:
	python src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL --start 2023-01-01 --end 2024-01-01

backtest-all:
	.\backtest_scripts\RUN_ALL_BASIC.bat

ingest:
	python src/run_ingestion.py

lint:
	python -m pylint src/ --disable=C0111,C0103

clean:
	-cmd /c "del /s /q *.pyc 2>nul"
	-cmd /c "for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d""
	-cmd /c "for /d /r . %d in (*.egg-info) do @if exist "%d" rd /s /q "%d""
	-cmd /c "for /d /r . %d in (.pytest_cache) do @if exist "%d" rd /s /q "%d""
