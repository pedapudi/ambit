"""The canonical in-memory representation every ambit stage reads and writes.

Ingestion normalizes whatever the user has (vectors, a parquet column, a model's
output) into one `EmbeddingSet`; projection, density, coverage and the
resolution/isotropy diagnostics all operate on it. Keeping a single contract is
what lets the three input tiers (embeddings / +dataset / +model) degrade
gracefully — downstream code never branches on where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class EmbeddingSet:
    X: np.ndarray                       # (n, d) float32 — the vectors
    ids: Optional[np.ndarray] = None    # (n,) stable identifiers (default: 0..n-1)
    labels: Optional[np.ndarray] = None # (n,) optional class/cluster labels
    meta: Optional[Any] = None          # optional pandas DataFrame of per-item metadata
    metric: str = "cosine"              # "cosine" | "euclidean"
    normalized: bool = False            # True once rows are L2-normalized
    source: Optional[str] = None        # provenance string for reporting

    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n, d); got shape {self.X.shape}")
        n, d = self.X.shape
        if n == 0 or d == 0:
            raise ValueError(f"empty embedding matrix: shape {self.X.shape}")
        if not np.isfinite(self.X).all():
            bad = int((~np.isfinite(self.X)).any(axis=1).sum())
            raise ValueError(f"{bad} row(s) contain NaN/Inf — clean or drop them before ingest")
        for name, arr in (("ids", self.ids), ("labels", self.labels)):
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != n_rows {n}")
        if self.ids is None:
            self.ids = np.arange(n)
        if self.metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean'; got {self.metric!r}")

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def normalize(self, eps: float = 1e-12) -> "EmbeddingSet":
        """Return an L2-normalized copy (cosine geometry becomes dot-product geometry)."""
        if self.normalized:
            return self
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        return EmbeddingSet(self.X / np.maximum(norms, eps), self.ids, self.labels,
                            self.meta, self.metric, True, self.source)

    def subsample(self, k: int, seed: int = 0) -> "EmbeddingSet":
        """Deterministic random subsample — the projected scatter/3-D views never draw
        more than a few thousand points, so large corpora are sampled for rendering
        while the scalar diagnostics still run over the full set."""
        if k >= self.n:
            return self
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.n, size=k, replace=False))
        meta = None if self.meta is None else self.meta.iloc[idx].reset_index(drop=True)
        labels = None if self.labels is None else self.labels[idx]
        return EmbeddingSet(self.X[idx], self.ids[idx], labels, meta,
                            self.metric, self.normalized, self.source)

    def __repr__(self) -> str:
        return (f"EmbeddingSet(n={self.n:,}, dim={self.dim}, metric={self.metric!r}, "
                f"normalized={self.normalized}, "
                f"labels={'yes' if self.labels is not None else 'no'}, "
                f"meta={'yes' if self.meta is not None else 'no'})")
