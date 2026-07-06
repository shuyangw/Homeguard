import pandas as pd
from src.data.acquisition.plugins import equity_index_yfinance as eq


def test_fetch_index_normalizes(monkeypatch):
    fake = pd.DataFrame({"Close": [4000.0, 4010.0]},
                        index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    monkeypatch.setattr(eq, "_download", lambda *a, **k: fake)
    out = eq.fetch_index("SPX", "2020-01-01", "2020-01-03", write=False)
    assert list(out.columns) == ["date", "close"]


def test_indices_map():
    assert eq.INDICES["SPX"] == "^GSPC"
    assert set(eq.INDICES) == {"SPX", "STOXX50E", "N225"}
