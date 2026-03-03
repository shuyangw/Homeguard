"""Tests for CSCMSignalLogger."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.trading.logging.cscm_signal_logger import CSCMSignalLogger


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def logger(log_dir):
    return CSCMSignalLogger(log_dir=log_dir)


class TestLogSignal:
    def test_creates_jsonl_with_signal(self, logger, log_dir):
        logger.log_signal(
            regime="bearish",
            btc_price=82450.0,
            btc_sma=85200.0,
            is_rebalance_day=False,
            reduce_exposure=True,
            exposure_pct=0.0,
            top_symbols=["SOL/USD", "ETH/USD", "LINK/USD"],
            momentum_scores={"SOL/USD": 0.15, "ETH/USD": 0.12, "LINK/USD": 0.08},
        )

        # Find the JSONL file
        cscm_dir = log_dir / "cscm"
        jsonl_files = list(cscm_dir.glob("cscm_signals_*.jsonl"))
        assert len(jsonl_files) == 1

        with open(jsonl_files[0]) as f:
            record = json.loads(f.readline())

        assert record["type"] == "signal"
        assert record["regime"] == "bearish"
        assert record["btc_price"] == 82450.0
        assert record["btc_sma"] == 85200.0
        assert record["reduce_exposure"] is True
        assert "SOL/USD" in record["top_symbols"]
        assert "timestamp" in record

    def test_appends_multiple_signals(self, logger, log_dir):
        logger.log_signal(
            regime="bearish", btc_price=82000.0, btc_sma=85000.0,
            is_rebalance_day=False, reduce_exposure=True, exposure_pct=0.0,
            top_symbols=[], momentum_scores={},
        )
        logger.log_signal(
            regime="bullish", btc_price=90000.0, btc_sma=85000.0,
            is_rebalance_day=False, reduce_exposure=False, exposure_pct=1.0,
            top_symbols=[], momentum_scores={},
        )

        cscm_dir = log_dir / "cscm"
        jsonl_files = list(cscm_dir.glob("cscm_signals_*.jsonl"))
        assert len(jsonl_files) == 1

        with open(jsonl_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 2

        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])
        assert record1["regime"] == "bearish"
        assert record2["regime"] == "bullish"


class TestLogRebalance:
    def test_includes_target_positions(self, logger, log_dir):
        targets = {"SOL/USD": 0.1429, "ETH/USD": 0.1429, "LINK/USD": 0.1429}
        logger.log_rebalance(
            regime="bullish",
            btc_price=92000.0,
            btc_sma=88000.0,
            top_symbols=["SOL/USD", "ETH/USD", "LINK/USD"],
            momentum_scores={"SOL/USD": 0.25, "ETH/USD": 0.18, "LINK/USD": 0.10},
            target_positions=targets,
        )

        cscm_dir = log_dir / "cscm"
        jsonl_files = list(cscm_dir.glob("cscm_signals_*.jsonl"))
        with open(jsonl_files[0]) as f:
            record = json.loads(f.readline())

        assert record["type"] == "rebalance"
        assert record["target_positions"] == targets
        assert record["regime"] == "bullish"


class TestErrorHandling:
    def test_handles_write_errors_gracefully(self, log_dir):
        # Use a path that will fail (file as directory)
        bad_dir = log_dir / "bad"
        bad_dir.mkdir(parents=True)
        # Create a file where the cscm directory should be
        blocker = bad_dir / "cscm"
        blocker.write_text("blocker")

        lgr = CSCMSignalLogger(log_dir=bad_dir)
        # Should not raise
        lgr.log_signal(
            regime="bearish", btc_price=80000.0, btc_sma=85000.0,
            is_rebalance_day=False, reduce_exposure=True, exposure_pct=0.0,
            top_symbols=[], momentum_scores={},
        )
