from datetime import date
import polars as pl


def test_build_carry_cache_writes_parquet(tmp_path, monkeypatch):
    import scripts.data.build_carry_cache as bcc

    monkeypatch.setattr(bcc, "carry_dir", lambda: tmp_path)

    def fake_compute_history(self, root, asset_class, start, end):
        return pl.DataFrame({"date": [date(2020, 1, 2), date(2020, 1, 3)],
                             "carry": [0.05, 0.06]})
    monkeypatch.setattr(bcc.CarryCalculator, "compute_history", fake_compute_history)

    written = bcc.build_carry_cache(["GC"], date(2020, 1, 1), date(2020, 1, 31))
    assert written == ["GC"]
    out = tmp_path / "GC.parquet"
    assert out.exists()
    df = pl.read_parquet(out)
    assert df.columns == ["date", "carry"]
    assert df.height == 2
