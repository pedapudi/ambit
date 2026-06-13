"""The shared render context. A `Ctx` is computed once from a `Scan` and passed to
every figure — projections (2-D / 3-D of the reservoir), the full-corpus
eigenspectrum, a random-pair cosine sample, the kNN graph, and labels (a provided
metadata column if present, else unsupervised clusters of the geometry itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import knn as knnmod
from . import metrics
from .project import project
from .scan import Scan


@dataclass
class Ctx:
    scan: Scan
    xy: np.ndarray                        # (m, 2) projected reservoir
    xyz: np.ndarray                       # (m, 3) projected reservoir
    eigs: np.ndarray                      # full-corpus covariance eigenvalues
    cos: np.ndarray                       # random-pair cosine sample (reservoir)
    knn_idx: Optional[np.ndarray] = None  # (m, k) neighbor indices over the reservoir
    knn_dist: Optional[np.ndarray] = None # (m, k) neighbor distances (1 - cos for cosine)
    labels: Optional[np.ndarray] = None   # (m,) group labels — provided or clustered
    labels_source: Optional[str] = None   # "provided" | "k-means (...)" | "hdbscan (...)" | None
    hub_skew: Optional[float] = None      # k-occurrence skewness over the reservoir kNN

    @property
    def es(self):
        return self.scan.sample


def build_ctx(sc: Scan, *, projector: str = "pca", pairs: int = 200_000,
              k: int = 10, clusters="auto", seed: int = 0) -> Ctx:
    """clusters: "auto"/True -> auto-label when no metadata column; an int -> force
    that many k-means clusters; False/None/0 -> no unsupervised labeling."""
    es = sc.sample.normalize()
    xy = project(es, 2, method=projector)
    xyz = project(es, 3, method=projector)
    cos = metrics.random_pair_cosine(es.X, n_pairs=pairs, normalized=True, seed=seed)
    try:
        knn_idx, knn_dist = knnmod.knn(es, k=k)
    except Exception:
        knn_idx = knn_dist = None

    labels = sc.sample.labels
    labels_source = "provided" if labels is not None else None
    if labels is None and clusters:
        try:
            from .cluster import cluster as _cluster
            kk = clusters if isinstance(clusters, int) and not isinstance(clusters, bool) and clusters > 0 else None
            method = clusters if isinstance(clusters, str) else "auto"
            labels, labels_source = _cluster(es.X, method=method, k=kk, seed=seed)
        except Exception:
            labels = labels_source = None

    hub_skew = metrics.hubness_skew(knn_idx) if knn_idx is not None else None
    return Ctx(sc, xy, xyz, sc.eigs, cos, knn_idx, knn_dist, labels, labels_source, hub_skew)
