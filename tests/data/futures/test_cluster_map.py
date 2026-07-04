import pytest
from src.data.futures.asset_class import CLUSTER, cluster_for
BROAD = ["ES","NQ","YM","ZT","ZF","ZN","TN","ZB","UB","6E","6J","6B","6A","6C","6S","6M","6N",
         "CL","BZ","NG","HO","RB","GC","SI","HG","PL","ZC","ZW","ZS","ZL","ZM","LE","HE"]
VALID = {"equity","rates","fx","energy","metals","grains","meats"}
def test_all_broad_roots_clustered():
    for r in BROAD: assert cluster_for(r) in VALID
def test_energy_split_from_metals_grains():
    assert cluster_for("CL")=="energy" and cluster_for("GC")=="metals"
    assert cluster_for("ZC")=="grains" and cluster_for("LE")=="meats"
    assert cluster_for("ES")=="equity" and cluster_for("ZN")=="rates" and cluster_for("6E")=="fx"
def test_unmapped_raises():
    with pytest.raises(KeyError): cluster_for("NOPE")
