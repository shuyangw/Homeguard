"""Expected values for futures validation checks.

Single source of truth for density ranges, value ranges, listing dates,
known gaps, known events, and per-section schemas. Type-hinted dicts
chosen over external YAML so refactoring tools work.
"""

from __future__ import annotations

from datetime import date

EXPECTED_DENSITY: dict[str, tuple[int, int]] = {
    # Equity index full + micros: near-24h Globex, very active
    "ES":  (800, 1000), "NQ":  (800, 1000), "YM":  (700, 950), "RTY": (700, 950),
    "MES": (800, 1000), "MNQ": (800, 1000), "M2K": (650, 900), "MYM": (650, 900),
    # Energy: near-24h
    "CL":  (800, 1000), "NG":  (700, 950),  "HO":  (600, 900), "RB":  (600, 900),
    "BZ":  (600, 900),  "MCL": (700, 950),  "MNG": (600, 900),
    # Metals: near-24h, recently very active
    "GC":  (900, 1200), "SI":  (700, 1000), "HG":  (600, 900), "PL":  (500, 800),
    "MGC": (700, 1000), "SIL": (500, 800),
    # Rates: high activity but not 24h
    "ZT":  (500, 750),  "ZF":  (550, 800),  "ZN":  (650, 900), "TN":  (300, 600),
    "ZB":  (500, 750),  "UB":  (300, 600),  "SR3": (300, 600), "SR1": (200, 500),
    "10Y": (200, 500),  "30Y": (15, 100),   "5YY": (10, 80),   "2YY": (25, 150),
    # FX: near-24h on majors
    "6E":  (500, 800),  "6J":  (500, 800),  "6B":  (300, 700), "6A":  (300, 700),
    "6C":  (300, 700),  "6S":  (200, 500),  "6N":  (200, 500), "6M":  (200, 500),
    # Ag: limited overnight
    "ZC":  (300, 700),  "ZS":  (300, 700),  "ZW":  (300, 700), "KE":  (200, 500),
    "ZL":  (300, 700),  "ZM":  (300, 700),  "LE":  (150, 400), "HE":  (150, 400),
    # Crypto: 24/7 but volume varies
    "BTC": (300, 800),  "MBT": (200, 700),  "ETH": (300, 800), "MET": (200, 700),
}

EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    "ES":  (600, 6500),
    "NQ":  (1700, 23000),
    "YM":  (8000, 50000),
    "RTY": (700, 2700),
    "GC":  (700, 3500),
    "SI":  (8, 60),
    "HG":  (1.5, 6.0),
    "PL":  (500, 1800),
    "CL":  (-50, 150),     # WTI went negative in April 2020
    "BZ":  (10, 150),
    "NG":  (1, 15),
    "ZN":  (105, 145),
    "ZB":  (100, 180),
    "10Y": (0, 8),         # Micro Yield: yield, not price
    "30Y": (0, 8),
    "5YY": (0, 8),
    "2YY": (0, 8),
    "6E":  (0.95, 1.55),
    "6J":  (60, 130),
    "BTC": (200, 200000),
    "ETH": (50, 7000),
}

LISTING_DATES: dict[str, date] = {
    # Symbols listed after dataset floor (2010-06-06).
    "MGC": date(2010, 10, 7),
    "SIL": date(2012, 8, 27),
    "TN":  date(2016, 1, 11),
    "RTY": date(2017, 7, 9),
    "BTC": date(2017, 12, 18),
    "SR3": date(2018, 5, 7),
    "SR1": date(2018, 5, 7),
    "MES": date(2019, 5, 6),
    "MNQ": date(2019, 5, 6),
    "M2K": date(2019, 5, 6),
    "MYM": date(2019, 5, 6),
    "ETH": date(2021, 2, 8),
    "MBT": date(2021, 5, 3),
    "MCL": date(2021, 7, 12),
    "MET": date(2021, 12, 6),
    "MNG": date(2022, 3, 28),
    "10Y": date(2022, 8, 15),
    "30Y": date(2022, 8, 15),
    "5YY": date(2022, 8, 15),
    "2YY": date(2022, 8, 15),
}

# (start_date, end_date, reason)
KNOWN_GAPS: list[tuple[date, date, str]] = [
    (date(2012, 10, 29), date(2012, 10, 31), "Hurricane Sandy NYSE close"),
]

# date -> (symbol, description)
KNOWN_EVENTS: dict[date, tuple[str, str]] = {
    date(2020, 3, 9):  ("ES", "circuit breaker / largest single-day drop in years"),
    date(2020, 4, 20): ("CL", "negative WTI oil prices"),
    date(2024, 8, 5):  ("ES", "yen carry trade unwind / major selloff"),
    date(2025, 4, 9):  ("ES", "tariff uncertainty selloff"),
}

# Expected schema per section: list of (column_name, polars_dtype_str).
# Empty `[]` means schema is large/varies and we only validate the
# timestamp column's existence and dtype.
EXPECTED_SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "futures_1min": [
        ("timestamp", "Datetime(time_unit='us', time_zone='UTC')"),
        ("open", "Float64"), ("high", "Float64"),
        ("low", "Float64"), ("close", "Float64"),
        ("volume", "UInt64"),
    ],
    "futures_1min_oi_roll": [
        ("timestamp", "Datetime(time_unit='us', time_zone='UTC')"),
        ("open", "Float64"), ("high", "Float64"),
        ("low", "Float64"), ("close", "Float64"),
        ("volume", "UInt64"),
    ],
    "futures_per_contract_1min": [
        ("timestamp", "Datetime(time_unit='us', time_zone='UTC')"),
        ("open", "Float64"), ("high", "Float64"),
        ("low", "Float64"), ("close", "Float64"),
        ("volume", "UInt64"), ("symbol", "String"),
    ],
    "futures_options_1min": [
        ("timestamp", "Datetime(time_unit='us', time_zone='UTC')"),
        ("open", "Float64"), ("high", "Float64"),
        ("low", "Float64"), ("close", "Float64"),
        ("volume", "UInt64"), ("symbol", "String"),
    ],
    "futures_definitions": [
        ("timestamp", "Datetime(time_unit='us', time_zone='UTC')"),
        ("raw_symbol", "String"),
        ("expiration", "Datetime(time_unit='ns', time_zone='UTC')"),
    ],
    "futures_statistics": [
        ("timestamp", "Datetime(time_unit='us', time_zone='UTC')"),
        ("symbol", "String"), ("price", "Float64"),
        ("stat_type", "UInt16"),
    ],
}
