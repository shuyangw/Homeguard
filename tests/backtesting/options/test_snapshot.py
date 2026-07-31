"""Tests for the snapshot-symmetry guard (spec v2 Section 1.3 rule 1).

Includes a NEGATIVE CONTROL: a lookahead-injecting truncation variant must be
REJECTED by the verifier. Without it the guard test could pass vacuously.
"""
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from src.backtesting.options.snapshot import (
    EASTERN_TZ,
    RTH_OPEN_ET,
    SNAPSHOT_TIME_ET,
    SnapshotLeakError,
    effective_cutoff_et,
    session_close_map,
    truncate_to_snapshot,
    verify_snapshot_truncation,
)

# 2024-07-03 is a registered NYSE half day (13:00 ET close).
EARLY_CLOSE_SESSION = date(2024, 7, 3)
NORMAL_SESSION = date(2024, 7, 5)


def _bars(session: date, first: str = "04:00", last: str = "19:59") -> pd.DataFrame:
    idx = pd.date_range(
        f"{session} {first}", f"{session} {last}", freq="1min", tz=EASTERN_TZ
    )
    n = len(idx)
    px = 100.0 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "timestamp": idx.tz_convert("UTC"),
            "open": px,
            "high": px + 0.05,
            "low": px - 0.05,
            "close": px,
            "volume": np.full(n, 1000),
        }
    )


def test_snapshot_minute_is_1545_and_not_reparameterized():
    assert SNAPSHOT_TIME_ET == time(15, 45)
    assert RTH_OPEN_ET == time(9, 30)


def test_cutoff_is_1545_on_a_normal_session():
    sched = session_close_map(NORMAL_SESSION, NORMAL_SESSION)
    assert effective_cutoff_et(NORMAL_SESSION, sched) == time(15, 45)


def test_cutoff_is_clamped_on_an_early_close():
    sched = session_close_map(EARLY_CLOSE_SESSION, EARLY_CLOSE_SESSION)
    cutoff = effective_cutoff_et(EARLY_CLOSE_SESSION, sched)
    assert cutoff < SNAPSHOT_TIME_ET
    # bar-START labelling: the last real bar of a 13:00 close is 12:59
    assert cutoff == time(12, 59)


def test_truncation_keeps_the_1545_bar_inclusive():
    out = truncate_to_snapshot(_bars(NORMAL_SESSION))
    last = out["ts_et"].max()
    assert last.time() == time(15, 45)


def test_truncation_drops_every_bar_after_the_cutoff():
    out = truncate_to_snapshot(_bars(NORMAL_SESSION))
    assert (out["ts_et"].dt.time <= SNAPSHOT_TIME_ET).all()
    verify_snapshot_truncation(out)


def test_truncation_drops_premarket_by_default():
    out = truncate_to_snapshot(_bars(NORMAL_SESSION))
    assert out["ts_et"].dt.time.min() == RTH_OPEN_ET


def test_truncation_excludes_padded_early_close_bars():
    """Half days are padded to 16:00 with stale quotes -- those must never
    become the mark."""
    out = truncate_to_snapshot(_bars(EARLY_CLOSE_SESSION))
    assert out["ts_et"].max().time() == time(12, 59)
    verify_snapshot_truncation(out)


def test_negative_control_verifier_rejects_a_lookahead_variant():
    """NEGATIVE CONTROL.

    A truncation that admits one extra minute is a lookahead leak. If the
    verifier accepted it, every truncation test above would be vacuous.
    """
    bars = _bars(NORMAL_SESSION)
    et = bars["timestamp"].dt.tz_convert(EASTERN_TZ)
    leaky = bars.assign(ts_et=et, session_date=et.dt.date)
    leaky = leaky[
        (leaky["ts_et"].dt.time >= RTH_OPEN_ET)
        & (leaky["ts_et"].dt.time <= time(15, 46))  # <-- the injected leak
    ]
    with pytest.raises(SnapshotLeakError):
        verify_snapshot_truncation(leaky)


def test_negative_control_verifier_rejects_a_padded_early_close_bar():
    bars = _bars(EARLY_CLOSE_SESSION)
    et = bars["timestamp"].dt.tz_convert(EASTERN_TZ)
    leaky = bars.assign(ts_et=et, session_date=et.dt.date)
    leaky = leaky[
        (leaky["ts_et"].dt.time >= RTH_OPEN_ET)
        & (leaky["ts_et"].dt.time <= SNAPSHOT_TIME_ET)
    ]
    with pytest.raises(SnapshotLeakError):
        verify_snapshot_truncation(leaky)


def test_multi_session_frame_is_truncated_per_session():
    bars = pd.concat(
        [_bars(EARLY_CLOSE_SESSION), _bars(NORMAL_SESSION)], ignore_index=True
    )
    out = truncate_to_snapshot(bars)
    by = out.groupby("session_date")["ts_et"].max()
    assert by[EARLY_CLOSE_SESSION].time() == time(12, 59)
    assert by[NORMAL_SESSION].time() == time(15, 45)
    verify_snapshot_truncation(out)


def _options_partition():
    from src.settings import get_local_storage_dir

    return (
        get_local_storage_dir() / "options" / "options_combined"
        / "root=SPY" / "year=2024" / "month=07" / "data.parquet"
    )


@pytest.mark.skipif(
    not _options_partition().exists(), reason="options_combined not on this host"
)
def test_disk_the_options_store_labels_bars_differently_from_the_equity_store():
    """MEASURED, not assumed. The two stores' conventions differ, which is why
    `canonical` clamps to the close minute inclusive and this module does not.

    A session carries 391 labels 09:30..16:00. The 09:30 label is universally
    zero-quoted (it is the pre-open state) yet carries volume; the 16:00 label
    has live quotes and no volume.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(_options_partition())
    t = pf.read_row_groups(
        [0], columns=["timestamp", "bid_close", "ask_close", "volume"]
    ).to_pandas()
    t["hhmm"] = t["timestamp"].str.slice(11, 16)
    labels = sorted(t["hhmm"].unique())
    assert len(labels) == 391
    assert labels[0] == "09:30" and labels[-1] == "16:00"

    open_bar = t[t["hhmm"] == "09:30"]
    assert ((open_bar["bid_close"] == 0) & (open_bar["ask_close"] == 0)).all()
    assert (open_bar["volume"] > 0).any()

    close_bar = t[t["hhmm"] == "16:00"]
    assert not ((close_bar["bid_close"] == 0) & (close_bar["ask_close"] == 0)).all()
    assert (close_bar["volume"] == 0).all()


def test_naive_timestamps_are_treated_as_eastern_wall_clock():
    bars = _bars(NORMAL_SESSION)
    bars["timestamp"] = bars["timestamp"].dt.tz_convert(EASTERN_TZ).dt.tz_localize(None)
    out = truncate_to_snapshot(bars)
    assert out["ts_et"].max().time() == time(15, 45)
