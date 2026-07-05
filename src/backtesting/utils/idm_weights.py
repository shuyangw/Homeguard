"""Data-free Instrument Diversification Multiplier (IDM) per-root div_mult weights.

Cluster risk weighting: equal risk budget across clusters present in the
universe, equal risk within each cluster. IDM uses a fixed intra-cluster
correlation of 0.5 and inter-cluster correlation of 0.0, capped at 2.5.
Parameter-free, deterministic, no I/O, no market data.
"""

import numpy as np

from src.data.futures.asset_class import cluster_for as _default_cluster_for

INTRA_CLUSTER_RHO = 0.5
IDM_CAP = 2.5


def compute_div_mult(
    universe: list[str],
    per_instrument_cap: float | None = None,
    cluster_fn=_default_cluster_for,
) -> dict[str, float]:
    """Return {symbol: div_mult} for every symbol in `universe`.

    `cluster_fn` maps a symbol to its cluster label (default: the futures
    asset-class map, preserving existing behavior). Raises KeyError if any
    symbol is unmapped.

    `per_instrument_cap`, if given, clips each symbol's div_mult to at most
    that value. Default None reproduces the uncapped output exactly.
    """
    clusters = [cluster_fn(sym) for sym in universe]
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
    # Cap is applied AFTER the N_scale median-pin, so it clips outliers without
    # disturbing the median-1.0 normalization of the uncapped values.
    if per_instrument_cap is not None:
        div_mult = np.minimum(div_mult, per_instrument_cap)

    return {root: float(dm) for root, dm in zip(universe, div_mult)}
