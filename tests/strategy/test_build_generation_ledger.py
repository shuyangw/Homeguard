import pytest

from scripts.strategy.build_generation_ledger import _GRADE, parse_rows

HEADER = "| # | name | capability | needs | spec | run | gate | notes |\n"


def _row(idx, grade):
    return f"| {idx} | Some idea | READY | data | spec | run | {grade} | notes |\n"


def test_known_grades_map_to_status():
    text = HEADER + _row(1, "-") + _row(2, "FAIL") + _row(3, "REJECT")
    rows = parse_rows(text)
    assert [r["status"] for r in rows] == ["OPEN", "TESTED-FAIL", "TESTED-DEAD"]


def test_unknown_grade_raises_instead_of_defaulting_to_open():
    """A silent default turned a tested-and-failed slot into an OPEN one and
    invited the generator to re-propose a dead spec. It must fail loudly."""
    text = HEADER + _row(20, "FAIL (probably)")
    with pytest.raises(ValueError, match="unrecognized gate grade"):
        parse_rows(text)


def test_cost_robust_fail_variants_both_recognized():
    """The tracker wrote 'FAIL (cost-robust)' while the map had it unspaced,
    which is the exact mismatch that mis-reported slot #20 as OPEN."""
    for grade in ("FAIL(cost-robust)", "FAIL (cost-robust)"):
        assert _GRADE[grade] == "TESTED-FAIL"


def test_weak_is_collapsed_to_fail():
    """WEAK is withheld on purpose: 'this nearly passed' invites the generator
    to aim at the near-miss."""
    rows = parse_rows(HEADER + _row(1, "WEAK"))
    assert rows[0]["status"] == "TESTED-FAIL"


def test_non_table_and_non_numeric_rows_ignored():
    text = HEADER + "|---|---|---|---|---|---|---|---|\n" + "prose line\n" + _row(7, "PASS")
    rows = parse_rows(text)
    assert [(r["id"], r["status"]) for r in rows] == [(7, "PASSED")]
