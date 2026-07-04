"""Data-free Instrument Diversification Multiplier (IDM) per-root div_mult weights.

Cluster risk weighting: equal risk budget across clusters present in the
universe, equal risk within each cluster. IDM uses a fixed intra-cluster
correlation of 0.5 and inter-cluster correlation of 0.0, capped at 2.5.
Parameter-free, deterministic, no I/O, no market data.
"""

import numpy as np

from src.data.futures.asset_class import cluster_for

INTRA_CLUSTER_RHO = 0.5
IDM_CAP = 2.5


def compute_div_mult(universe: list[str]) -> dict[str, float]:
    """Return {root: div_mult} for every root in `universe`.

    Raises KeyError if any root is unmapped in `cluster_for`.
    """
    clusters = [cluster_for(root) for root in universe]
    clusters_present = set(clusters)
    n_clusters = len(clusters_present)
    cluster_counts = {c: clusters.count(c) for c in clusters_present}

    weights = np.array(
        [(1.0 / n_clusters) / cluster_counts[c] for c in clusters]
    )

    n = len(universe)
    corr = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j and clusters[i] == clusters[j]:
                corr[i, j] = INTRA_CLUSTER_RHO

    portfolio_variance = weights @ corr @ weights
    idm = min(1.0 / np.sqrt(portfolio_variance), IDM_CAP)

    raw_dm = weights * idm
    n_scale = 1.0 / np.median(raw_dm)
    div_mult = raw_dm * n_scale

    return {root: float(dm) for root, dm in zip(universe, div_mult)}
