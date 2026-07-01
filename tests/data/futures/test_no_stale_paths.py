"""Guard: no module may construct a path from the pre-consolidation flat futures dirs.

Scoped to PATH-CONSTRUCTION usage only (a quoted futures dir literal preceded by
a path-join `/`), not the EXPECTED_SCHEMAS dict label keys in expectations.py or
display/error-message text elsewhere -- those are legitimate and unrelated to the
on-disk layout consolidation.
"""
import subprocess


def test_no_stale_futures_path_construction():
    # Match a quoted futures dir literal used in a path join (root / "futures_1min"),
    # NOT schema-label dict keys or error-message text.
    res = subprocess.run(
        ["git", "grep", "-nE",
         r'/ *"futures_(1min|per_contract_1min|per_contract_daily|statistics|definitions)"',
         "--", "src/"],
        capture_output=True, text=True,
    )
    # git grep exits 1 (no matches) when clean; 0 (matches) when stale refs remain
    assert res.returncode == 1, f"stale futures path construction remains:\n{res.stdout}"
