"""The shared render context. A `Ctx` is computed once from a `Scan` and passed to
every figure — projections (2-D / 3-D of the reservoir), the full-corpus
eigenspectrum, a random-pair cosine sample, and the kNN graph over the reservoir.
Figures read what they need and emit token-colored SVG, so a figure never
re-derives the heavy structures.
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
    xy: np.ndarray                       # (m, 2) projected reservoir
    xyz: np.ndarray                      # (m, 3) projected reservoir
    eigs: np.ndarray                     # full-corpus covariance eigenvalues
    cos: np.ndarray                      # random-pair cosine sample (reservoir)
    knn_idx: Optional[np.ndarray] = None # (m, k) neighbor indices over the reservoir
    knn_dist: Optional[np.ndarray] = None# (m, k) neighbor distances (1 - cos for cosine)

    @property
    def es(self):
        return self.scan.sample

    @property
    def labels(self):
        return self.scan.sample.labels


def build_ctx(sc: Scan, *, projector: str = "pca", pairs: int = 200_000,
              k: int = 10, seed: int = 0) -> Ctx:
    es = sc.sample.normalize()
    xy = project(es, 2, method=projector)
    xyz = project(es, 3, method=projector)
    cos = metrics.random_pair_cosine(es.X, n_pairs=pairs, normalized=True, seed=seed)
    try:
        knn_idx, knn_dist = knnmod.knn(es, k=k)
    except Exception:
        knn_idx = knn_dist = None  # kNN figures degrade gracefully if no backend
    return Ctx(sc, xy, xyz, sc.eigs, cos, knn_idx, knn_dist)
