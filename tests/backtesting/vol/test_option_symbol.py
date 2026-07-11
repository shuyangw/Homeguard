from src.backtesting.vol.option_symbol import parse_option_symbol


def test_parse_es_call():
    o = parse_option_symbol("ESH2 C4725", ref_year=2022)
    assert o.root == "ES" and o.right == "C" and o.strike == 4725.0
    assert o.expiry_month == 3 and o.expiry_year == 2022


def test_parse_nq_put():
    o = parse_option_symbol("NQH2 P1577", ref_year=2022)
    assert o.root == "NQ" and o.right == "P" and o.strike == 1577.0 and o.expiry_month == 3 and o.expiry_year == 2022


def test_non_option_returns_none():
    assert parse_option_symbol("ESH2") is None
    assert parse_option_symbol("ES-NQ") is None
    assert parse_option_symbol("") is None


def test_ref_year_resolves_pre_2020():
    # H5 on a 2015 trading day -> March 2015, not 2025
    o = parse_option_symbol("ESH5 C2000", ref_year=2015)
    assert o.expiry_year == 2015 and o.expiry_month == 3
