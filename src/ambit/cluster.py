"""Unsupervised labeling — ambit's own grouping of the embedding space, no model
required. When the dataset ships no label column, the report still gets meaningful
groups from the geometry itself (the clusters ARE the structure of the space).

HDBSCAN if available (density-based, picks the count itself, marks noise);
otherwise k-means with a silhouette-picked k. Naming the clusters (semantic
labels) is a separate, optional step that DOES need a model — this does not.
"""

from __future__ import annotations

import numpy as np


def cluster(X, *, method: str = "auto", k=None, k_range=(3, 12), seed: int = 0):
    """Return (labels, source). X should be L2-normalized (cosine geometry)."""
    X = np.ascontiguousarray(X, dtype=np.float32)

    if method in ("auto", "hdbscan"):
        try:
            import hdbscan
            mcs = max(5, len(X) // 50)
            lab = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(X)
            n = len({int(v) for v in lab if v >= 0})
            if n >= 2:
                return lab, f"hdbscan (k={n})"
            if method == "hdbscan":
                return lab, "hdbscan (k<2)"
        except ImportError:
            if method == "hdbscan":
                raise

    from sklearn.cluster import KMeans
    if k is not None:
        lab = KMeans(n_clusters=int(k), n_init=4, random_state=seed).fit_predict(X)
        return lab, f"k-means (k={int(k)})"

    from sklearn.metrics import silhouette_score
    lo, hi = k_range
    hi = min(hi, len(X) - 1)
    best = None
    for kk in range(max(2, lo), max(3, hi) + 1):
        lab = KMeans(n_clusters=kk, n_init=4, random_state=seed).fit_predict(X)
        try:
            s = silhouette_score(X, lab, sample_size=min(2000, len(X)), random_state=seed)
        except Exception:
            s = -1.0
        if best is None or s > best[0]:
            best = (s, kk, lab)
    return best[2], f"k-means (k={best[1]}, silhouette {best[0]:.2f})"
