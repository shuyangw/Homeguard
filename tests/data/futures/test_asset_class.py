import pytest
from src.data.futures.asset_class import ASSET_CLASS, asset_class_for

BROAD = ["ES", "NQ", "YM", "ZT", "ZF", "ZN", "TN", "ZB", "UB",
         "6E", "6J", "6B", "6A", "6C", "6S", "6M", "6N",
         "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "HG", "PL",
         "ZC", "ZW", "ZS", "ZL", "ZM", "LE", "HE"]
VALID = {"equity_index", "fx", "bond", "commodity"}


def test_every_broad_root_mapped_to_valid_class():
    for r in BROAD:
        assert asset_class_for(r) in VALID


def test_spot_check_classes():
    assert asset_class_for("ES") == "equity_index"
    assert asset_class_for("6E") == "fx"
    assert asset_class_for("ZN") == "bond"
    assert asset_class_for("CL") == "commodity"
    assert asset_class_for("GC") == "commodity"


def test_unmapped_root_raises():
    with pytest.raises(KeyError):
        asset_class_for("NOPE")
