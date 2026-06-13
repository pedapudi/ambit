"""One-pass streaming scan — the ingestion entry point for corpora too large to
hold in RAM. A single pass over the source computes the EXACT full-set covariance
and norm statistics (so effective rank / variance are over all N), while keeping a
bounded reservoir sample of vectors for the projection, scatter, 3-D and cosine
layers (which can't draw a million points anyway).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .source import iter_chunks
from .types import EmbeddingSet


@dataclass
class Scan:
    n: int                  # rows over the full corpus
    dim: int
    cov: np.ndarray         # (d, d) covariance over the full corpus
    mean: np.ndarray        # (d,) mean over the full corpus
    norm_mean: float
    norm_std: float
    sample: EmbeddingSet    # bounded reservoir sample (the visual working set)
    source: str
    metric: str = "cosine"

    @property
    def eigs(self) -> np.ndarray:
        return metrics.eigs_from_cov(self.cov)


def scan(source, *, sample: int = 20_000, seed: int = 0, embedding_col=None,
         id_col=None, label_col=None, metric: str = "cosine",
         batch_rows: int = 50_000) -> Scan:
    rng = np.random.default_rng(seed)
    n = 0
    dim = None
    sumv = XtX = None
    res = res_ids = res_lab = None
    norm_sum = norm_sqsum = 0.0

    for ch in iter_chunks(source, embedding_col=embedding_col, id_col=id_col,
                          label_col=label_col, batch_rows=batch_rows):
        Xc = np.ascontiguousarray(ch.X, dtype=np.float64)
        m, d = Xc.shape
        if dim is None:
            dim = d
            sumv = np.zeros(d)
            XtX = np.zeros((d, d))
            res = np.zeros((sample, d), dtype=np.float32)
            res_ids = np.empty(sample, dtype=object) if ch.ids is not None else None
            res_lab = np.empty(sample, dtype=object) if ch.labels is not None else None
        elif d != dim:
            raise ValueError(f"inconsistent embedding dim: {d} != {dim}")

        sumv += Xc.sum(0)
        XtX += Xc.T @ Xc                       # float64 scatter accumulation
        nr = np.linalg.norm(Xc, axis=1)
        norm_sum += float(nr.sum())
        norm_sqsum += float((nr * nr).sum())

        # reservoir sample: fill empty slots first, then replace with decaying probability
        filled = min(n, sample)
        take = min(sample - filled, m)
        if take > 0:
            res[filled:filled + take] = ch.X[:take]
            if res_ids is not None: res_ids[filled:filled + take] = ch.ids[:take]
            if res_lab is not None: res_lab[filled:filled + take] = ch.labels[:take]
        if m > take:
            rest = ch.X[take:]
            t = n + take + np.arange(1, (m - take) + 1)     # 1-based global positions (> sample)
            j = rng.integers(0, t)
            acc = j < sample
            slots = j[acc]
            res[slots] = rest[acc]
            if res_ids is not None: res_ids[slots] = ch.ids[take:][acc]
            if res_lab is not None: res_lab[slots] = ch.labels[take:][acc]
        n += m

    if n == 0:
        raise ValueError("empty source: no rows ingested")
    mean = sumv / n
    cov = (XtX - n * np.outer(mean, mean)) / max(1, n - 1)
    keep = min(n, sample)
    es = EmbeddingSet(res[:keep],
                      ids=None if res_ids is None else res_ids[:keep],
                      labels=None if res_lab is None else res_lab[:keep],
                      metric=metric, source=str(source))
    norm_mean = norm_sum / n
    norm_std = float(np.sqrt(max(0.0, norm_sqsum / n - norm_mean ** 2)))
    return Scan(n, dim, cov, mean, norm_mean, norm_std, es, str(source), metric)
