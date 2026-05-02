"""Tests for src.features.normalizers."""
import numpy as np
import pandas as pd
import pytest


def test_module_importable():
    from src.features import normalizers  # noqa: F401


class TestLogTransform:
    def test_inverse_property(self):
        from src.features import log_transform
        rng = np.random.default_rng(0)
        s = pd.Series(rng.uniform(0.1, 100.0, 50))
        recovered = np.exp(log_transform(s))
        np.testing.assert_allclose(recovered.to_numpy(), s.to_numpy(), atol=1e-10)

    def test_non_positive_produces_nan(self):
        from src.features import log_transform
        s = pd.Series([1.0, 0.0, -1.0, 2.0])
        out = log_transform(s)
        assert np.isnan(out.iloc[1])
        assert np.isnan(out.iloc[2])
        assert np.isclose(out.iloc[0], 0.0)
        assert np.isclose(out.iloc[3], np.log(2.0))

    def test_nan_propagates(self):
        from src.features import log_transform
        s = pd.Series([1.0, np.nan, 2.0])
        out = log_transform(s)
        assert np.isnan(out.iloc[1])
        assert not np.isnan(out.iloc[0])
        assert not np.isnan(out.iloc[2])

    def test_empty_series(self):
        from src.features import log_transform
        out = log_transform(pd.Series([], dtype=float))
        assert len(out) == 0
        assert isinstance(out, pd.Series)

    def test_index_preserved(self):
        from src.features import log_transform
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        out = log_transform(s)
        assert out.index.equals(idx)
        assert len(out) == len(s)
