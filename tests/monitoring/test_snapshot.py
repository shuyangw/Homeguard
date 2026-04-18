"""Tests for snapshot writer/reader."""

import json
import os
import tempfile
import time

from src.monitoring.registry import MetricsRegistry
from src.monitoring.snapshot import SnapshotWriter, read_snapshot


class TestSnapshotWriter:

    def test_write_and_read_roundtrip(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_portfolio(100000.0, 50000.0, 150000.0, 'alpaca')
        reg.update_market_open(True)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SnapshotWriter(reg, snapshot_dir=tmpdir, interval_seconds=1)
            writer.write_once()

            snapshot = read_snapshot(tmpdir)
            assert snapshot is not None
            assert snapshot['strategy'] == 'omr'
            assert 'hg_portfolio_equity_usd' in snapshot['gauges']
            assert snapshot['timestamp'] > 0

    def test_staleness_detection(self):
        reg = MetricsRegistry(strategy='omr')

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SnapshotWriter(reg, snapshot_dir=tmpdir, interval_seconds=1)
            writer.write_once()

            snapshot = read_snapshot(tmpdir, max_age_seconds=9999)
            assert snapshot is not None

            snapshot_stale = read_snapshot(tmpdir, max_age_seconds=0)
            assert snapshot_stale is None

    def test_snapshot_file_created(self):
        reg = MetricsRegistry(strategy='ramp')

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SnapshotWriter(reg, snapshot_dir=tmpdir, interval_seconds=1)
            writer.write_once()

            files = os.listdir(tmpdir)
            assert any(f.startswith('ramp_snapshot') for f in files)
