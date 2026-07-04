"""Root -> asset_class map for carry (the strings CarryCalculator.compute accepts)."""

ASSET_CLASS: dict[str, str] = {
    # equity_index
    "ES": "equity_index", "NQ": "equity_index", "YM": "equity_index", "RTY": "equity_index",
    "M2K": "equity_index", "MES": "equity_index", "MNQ": "equity_index", "MYM": "equity_index",
    # fx
    "6A": "fx", "6B": "fx", "6C": "fx", "6E": "fx", "6J": "fx", "6M": "fx", "6N": "fx", "6S": "fx",
    # bond
    "ZT": "bond", "ZF": "bond", "ZN": "bond", "TN": "bond", "ZB": "bond", "UB": "bond",
    "10Y": "bond", "2YY": "bond", "5YY": "bond", "30Y": "bond", "SR1": "bond", "SR3": "bond",
    # commodity
    "CL": "commodity", "BZ": "commodity", "NG": "commodity", "HO": "commodity", "RB": "commodity",
    "MCL": "commodity", "MNG": "commodity", "GC": "commodity", "SI": "commodity", "HG": "commodity",
    "PL": "commodity", "MGC": "commodity", "SIL": "commodity", "MET": "commodity",
    "ZC": "commodity", "ZW": "commodity", "ZS": "commodity", "ZL": "commodity", "ZM": "commodity",
    "KE": "commodity", "LE": "commodity", "HE": "commodity",
}


def asset_class_for(root: str) -> str:
    """Return the carry asset_class for `root`; raise KeyError if unmapped."""
    return ASSET_CLASS[root]


CLUSTER: dict[str, str] = {
    "ES": "equity", "NQ": "equity", "YM": "equity", "RTY": "equity",
    "M2K": "equity", "MES": "equity", "MNQ": "equity", "MYM": "equity",
    "ZT": "rates", "ZF": "rates", "ZN": "rates", "TN": "rates", "ZB": "rates", "UB": "rates",
    "10Y": "rates", "2YY": "rates", "5YY": "rates", "30Y": "rates", "SR1": "rates", "SR3": "rates",
    "6A": "fx", "6B": "fx", "6C": "fx", "6E": "fx", "6J": "fx", "6M": "fx", "6N": "fx", "6S": "fx",
    "CL": "energy", "BZ": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "MCL": "energy", "MNG": "energy",
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals", "MGC": "metals",
    "SIL": "metals", "MET": "metals",
    "ZC": "grains", "ZW": "grains", "ZS": "grains", "ZL": "grains", "ZM": "grains", "KE": "grains",
    "LE": "meats", "HE": "meats",
}


def cluster_for(root: str) -> str:
    return CLUSTER[root]
