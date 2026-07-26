"""The trial count must never quietly shrink.

get_campaign_trial_distribution() swallowed every exception and returned the
static 40-trial baseline. A read-only query was calling init_db(), which opens
READ-WRITE, so anything holding the registry -- a concurrent session, Dropbox's
indexer, or an earlier read-only connection in the same process -- silently
dropped N from 141 to 40 and the deflated bar from 1.1372 to 0.7331. Nothing in
the run or its report would say so.
"""
import duckdb
import pytest

from src.backtesting.statistics.dsr import expected_max_sharpe
from src.backtesting.walkforward_common import (CAMPAIGN_CUMULATIVE_TRIALS,
                                                get_campaign_trial_distribution)
from src.experiments.registry import DEFAULT_DB_PATH


@pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="no local registry")
def test_reads_the_full_count_while_another_connection_holds_the_db():
    baseline_n, _ = get_campaign_trial_distribution()
    con = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        held_n, held_sharpes = get_campaign_trial_distribution()
    finally:
        con.close()
    assert held_n == baseline_n
    assert held_n > CAMPAIGN_CUMULATIVE_TRIALS, "fell back to the static baseline"


@pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="no local registry")
def test_the_bar_does_not_soften_under_contention():
    _, base = get_campaign_trial_distribution()
    n_base = len(base)
    con = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        n_held, held = get_campaign_trial_distribution()
    finally:
        con.close()
    assert len(held) == n_base
    assert expected_max_sharpe(held, n_held) == pytest.approx(
        expected_max_sharpe(base, n_held))


def test_missing_registry_falls_back_loudly(tmp_path, monkeypatch):
    """A genuine failure may still fall back, but it must not do so in silence."""
    import src.backtesting.walkforward_common as wf
    errors = []
    monkeypatch.setattr(wf.logger, "error", errors.append)
    n, sharpes = get_campaign_trial_distribution(db_path=tmp_path / "nope" / "x.duckdb")
    assert n == CAMPAIGN_CUMULATIVE_TRIALS
    assert len(errors) == 1 and "trial count" in errors[0].lower()
