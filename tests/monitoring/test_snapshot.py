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

    def test_strategy_filter(self):
        """read_snapshot(strategy='omr') must not return another strategy's snapshot."""
        omr_reg = MetricsRegistry(strategy='omr')
        ramp_reg = MetricsRegistry(strategy='ramp')

        with tempfile.TemporaryDirectory() as tmpdir:
            SnapshotWriter(omr_reg, snapshot_dir=tmpdir).write_once()
            time.sleep(0.01)  # ensure ramp has newer mtime
            SnapshotWriter(ramp_reg, snapshot_dir=tmpdir).write_once()

            # Unfiltered: latest is ramp
            assert read_snapshot(tmpdir)['strategy'] == 'ramp'
            # Filtered: picks omr even though ramp is newer
            assert read_snapshot(tmpdir, strategy='omr')['strategy'] == 'omr'
            # Missing strategy: None
            assert read_snapshot(tmpdir, strategy='cscm') is None

    def test_corrupt_snapshot_returns_none(self):
        """A malformed JSON file must return None, not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = os.path.join(tmpdir, 'broken_snapshot.json')
            with open(bad_file, 'w') as f:
                f.write('{not valid json')

            assert read_snapshot(tmpdir) is None
