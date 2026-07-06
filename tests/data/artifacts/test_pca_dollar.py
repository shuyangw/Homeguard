import numpy as np
import pandas as pd
from src.data.artifacts.pca_dollar import dollar_factor


def test_pc1_captures_common_move():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 1, 300)
    idx = pd.date_range("2020-01-01", periods=300)
    df = pd.DataFrame({
        "EURUSD": common + rng.normal(0, 0.1, 300),
        "GBPUSD": common + rng.normal(0, 0.1, 300),
        "AUDUSD": common + rng.normal(0, 0.1, 300),
    }, index=idx)
    pc1, resid = dollar_factor(df)
    assert len(pc1) == 300
    assert resid.shape == df.shape
    # residual variance is far smaller than raw variance once PC1 removed
    assert resid.var().mean() < df.var().mean()
