"""Tests for compute_lifetime_realized_pnl in src.utils.trading_logger.

Covers: multi-day aggregation, per-strategy filtering, mtime-based caching,
malformed lines, missing log dir, missing pnl_dollars, non-exit rows.
"""

import json
import os
import time
from pathlib import Path

import pytest

from src.utils import trading_logger
from src.utils.trading_logger import compute_lifetime_realized_pnl


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty cache to avoid cross-test bleed."""
    trading_logger._LIFETIME_PNL_CACHE.clear()
    yield
    trading_logger._LIFETIME_PNL_CACHE.clear()


def test_returns_zero_when_log_dir_missing(tmp_path):
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path / 'nope')) == 0.0


def test_returns_zero_when_no_files(tmp_path):
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 0.0


def test_sums_exits_for_strategy_across_multiple_files(tmp_path):
    _write_jsonl(tmp_path / 'trades_20260427.jsonl', [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 100.0},
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': -25.5},
    ])
    _write_jsonl(tmp_path / 'trades_20260428.jsonl', [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 50.0},
    ])
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == pytest.approx(124.5)


def test_filters_by_strategy(tmp_path):
    _write_jsonl(tmp_path / 'trades_20260427.jsonl', [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 100.0},
        {'strategy': 'cscm', 'trade_type': 'exit', 'pnl_dollars': 999.0},
        {'strategy': 'omr',  'trade_type': 'exit', 'pnl_dollars': -50.0},
    ])
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 100.0
    assert compute_lifetime_realized_pnl('cscm', str(tmp_path)) == 999.0
    assert compute_lifetime_realized_pnl('omr',  str(tmp_path)) == -50.0
    assert compute_lifetime_realized_pnl('mp',   str(tmp_path)) == 0.0


def test_skips_non_exit_rows(tmp_path):
    _write_jsonl(tmp_path / 'trades_20260427.jsonl', [
        {'strategy': 'ramp', 'trade_type': 'entry', 'pnl_dollars': 999.0},  # ignored
        {'strategy': 'ramp', 'trade_type': 'exit',  'pnl_dollars': 100.0},
    ])
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 100.0


def test_skips_rows_with_null_pnl(tmp_path):
    """Entries log_exit writes when entry_price is unknown have pnl_dollars=None."""
    _write_jsonl(tmp_path / 'trades_20260427.jsonl', [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': None},
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 50.0},
    ])
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 50.0


def test_skips_malformed_lines(tmp_path):
    log = tmp_path / 'trades_20260427.jsonl'
    with open(log, 'w', encoding='utf-8') as f:
        f.write('not-json\n')
        f.write(json.dumps({'strategy': 'ramp', 'trade_type': 'exit',
                            'pnl_dollars': 100.0}) + '\n')
        f.write('{"incomplete":\n')
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 100.0


def test_cache_hit_when_mtime_unchanged(tmp_path, monkeypatch):
    log = tmp_path / 'trades_20260427.jsonl'
    _write_jsonl(log, [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 100.0},
    ])
    # First call populates cache.
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 100.0
    cached = trading_logger._LIFETIME_PNL_CACHE[str(log)]
    # Second call: monkeypatch the per-file scan to fail. If we're hitting the
    # cache the file is never re-read so the patch never fires.
    def _explode(*args, **kwargs):
        raise AssertionError('cache miss -- file was re-read despite unchanged mtime')
    monkeypatch.setattr(trading_logger, '_sum_pnl_per_strategy_in_file', _explode)
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 100.0
    assert trading_logger._LIFETIME_PNL_CACHE[str(log)] == cached


def test_cache_invalidates_when_file_appended(tmp_path):
    log = tmp_path / 'trades_20260427.jsonl'
    _write_jsonl(log, [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 100.0},
    ])
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 100.0
    # Append a new exit. Bump mtime explicitly because some filesystems have
    # 1s mtime resolution and a fast append within the same second won't tick.
    with open(log, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'strategy': 'ramp', 'trade_type': 'exit',
                            'pnl_dollars': 50.0}) + '\n')
    new_mtime = log.stat().st_mtime + 1
    os.utime(log, (new_mtime, new_mtime))
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 150.0


def test_ignores_non_trades_files(tmp_path):
    """Glob is `trades_*.jsonl` -- other files in the dir must be ignored."""
    _write_jsonl(tmp_path / 'trades_20260427.jsonl', [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 100.0},
    ])
    _write_jsonl(tmp_path / 'executions_20260427.log', [
        {'strategy': 'ramp', 'trade_type': 'exit', 'pnl_dollars': 99999.0},
    ])
    (tmp_path / 'random.txt').write_text('not a trade log')
    assert compute_lifetime_realized_pnl('ramp', str(tmp_path)) == 100.0
