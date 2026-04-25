"""Tests for the CLI subcommands. Uses fixture jsonl + golden text."""
import io
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

from src.trading.decision_log import cli
from src.trading.decision_log.writer import append
from src.trading.decision_log.record import DecisionRecord


FIXTURES = Path(__file__).parent / "fixtures"


def _seed_from_fixture(tmp_decisions_dir, fixture_name):
    """Copy fixture jsonl into the temp decisions dir as today's record."""
    src = FIXTURES / fixture_name
    text = src.read_text()
    rec = DecisionRecord.from_jsonl_line(text)
    append(rec)


class TestShow:
    def test_show_latest_ramp_matches_golden(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["show", "ramp"])
        assert rc == 0
        out = capsys.readouterr().out
        expected = (FIXTURES / "expected_show_ramp_clean.txt").read_text()
        # Compare line-by-line for clearer diffs
        assert out.strip().split("\n") == expected.strip().split("\n")

    def test_show_with_json_flag_emits_raw_json(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["show", "ramp", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        loaded = json.loads(out)
        assert loaded["decision_id"] == "7c2f9d"

    def test_show_no_records_returns_nonzero(self, tmp_decisions_dir, capsys):
        rc = cli.main(["show", "ramp"])
        assert rc != 0


class TestList:
    def test_list_summary_table(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["list", "ramp", "--days", "7"])
        assert rc == 0
        out = capsys.readouterr().out
        # Table header
        assert "DATE" in out and "TRIGGER" in out and "REGIME" in out
        # The seed record's regime should appear
        assert "WEAK_BULL" in out
