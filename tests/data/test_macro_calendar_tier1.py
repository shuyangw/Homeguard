import datetime as dt

from src.data.macro_calendar_tier1 import load_tier1_rules, _expand_rule_dates


def test_load_tier1_rules_has_eur_and_gbp():
    rules = load_tier1_rules()
    names = {r["name"] for r in rules}
    assert {"EZ_FLASH_CPI", "UK_CPI", "ECB_DECISION", "BOE_DECISION"} <= names
    assert {r["currency"] for r in rules} == {"EUR", "GBP"}


def test_expand_nth_weekday():
    rule = {"cadence": "monthly:nth-weekday:3:WED"}
    dates = _expand_rule_dates(rule, dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    # 3rd Wednesday of Jan/Feb/Mar 2024
    assert dates == [dt.date(2024, 1, 17), dt.date(2024, 2, 21), dt.date(2024, 3, 20)]


def test_expand_month_end_business_day():
    rule = {"cadence": "monthly:month-end-business-day"}
    dates = _expand_rule_dates(rule, dt.date(2024, 3, 1), dt.date(2024, 3, 31))
    assert dates == [dt.date(2024, 3, 29)]  # 31st is Sun, 30th Sat -> Fri 29th


def test_expand_business_day_n():
    rule = {"cadence": "monthly:business-day:1"}
    dates = _expand_rule_dates(rule, dt.date(2024, 6, 1), dt.date(2024, 6, 30))
    assert dates == [dt.date(2024, 6, 3)]  # Jun 1 Sat, 2 Sun -> 1st biz day Mon 3rd


def test_expand_quarterly_only_anchor_months():
    rule = {"cadence": "quarterly:nth-weekday:5:TUE:1"}
    dates = _expand_rule_dates(rule, dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    # anchor month 1 -> Jan, Apr, Jul, Oct only
    assert [d.month for d in dates] == [1, 4, 7, 10]


def test_expand_from_cb_decisions_rule_returns_empty():
    rule = {"cadence": None, "from_cb_decisions": "ECB"}
    assert _expand_rule_dates(rule, dt.date(2024, 1, 1), dt.date(2024, 12, 31)) == []
