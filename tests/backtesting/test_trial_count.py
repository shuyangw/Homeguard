from src.backtesting.walkforward_common import CAMPAIGN_CUMULATIVE_TRIALS


def test_trial_count_is_honest_cumulative():
    # The campaign has run dozens of pre-registered trials (SP-A/E/B/C + sweep +
    # baselines). The deflation count must reflect that, not 1.
    assert CAMPAIGN_CUMULATIVE_TRIALS >= 40


def test_old_name_removed():
    import src.backtesting.walkforward_common as wc
    assert not hasattr(wc, "TRIAL_COUNT_PARAMETER_FREE"), \
        "old =1 constant must be renamed, not left as an alias"
