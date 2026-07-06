import pandas as pd
from src.data.acquisition.plugins import oil_yfinance


def test_normalize_shape(monkeypatch):
    fake = pd.DataFrame({"Close": [70.0, 71.0]},
                        index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    monkeypatch.setattr(oil_yfinance, "_download", lambda *a, **k: fake)
    out = oil_yfinance.fetch_brent("2020-01-01", "2020-01-03", write=False)
    assert list(out.columns) == ["date", "close"]
    assert len(out) == 2
