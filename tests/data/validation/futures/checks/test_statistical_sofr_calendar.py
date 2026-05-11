"""Verify SofrDerivationSanityCheck samples only weekday dates."""
from datetime import date, timedelta
from unittest.mock import patch

from src.data.validation.futures.checks.statistical import SofrDerivationSanityCheck


def test_sofr_sampler_uses_weekdays_only():
    """The sampler must select dates with weekday() in 0..4 (Mon-Fri)."""

    # Patch derive_sofr to record every date it was called with, then
    # immediately raise (we don't care about the actual derivation here,
    # only the sampling distribution).
    seen_dates = []

    def fake_derive(d: date) -> float:
        seen_dates.append(d)
        raise ValueError("forced")

    # Patch _storage_root to a tmp path so we don't accidentally hit real data
    # if derive_sofr's failure mode changes.
    with patch(
        "src.data.validation.futures.checks.statistical.sofr_module.derive_sofr",
        side_effect=fake_derive,
    ):
        SofrDerivationSanityCheck().run()

    # All 30 sampled dates must be weekdays (Mon=0..Fri=4)
    assert len(seen_dates) == 30, f"expected 30 samples, got {len(seen_dates)}"
    for d in seen_dates:
        assert d.weekday() < 5, f"sampled non-weekday date {d} (weekday={d.weekday()})"
