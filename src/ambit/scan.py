"""One-pass streaming scan — the ingestion entry point for corpora too large to
hold in RAM. A single pass computes the covariance + norm stats and keeps a bounded
reservoir sample for the visual layers.

Two knobs for 3-5M+:
  - device="cuda"/"auto"/"mps" routes the covariance accumulation onto a torch
    device (see accel.py); device="cpu" (default) uses numpy with float32 grams.
  - approx=N stops after ~N rows, estimating the covariance/spectrum from that
    sample instead of the exact full pass (rank/variance converge fast in n, so a
    few hundred-k sample is close to exact at a fraction of the cost). The headline
    item count still reflects the true corpus size when it is cheaply known.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import metrics
from .source import iter_chunks
from .types import EmbeddingSet


@dataclass
class Scan:
    n: int                  # corpus rows (true total if cheaply known, else scanned)
    dim: int
    cov: np.ndarray         # (d, d) covariance over the scanned rows
    mean: np.ndarray
    norm_mean: float
    norm_std: float
    sample: EmbeddingSet    # bounded reservoir sample (the visual working set)
    source: str
    metric: str = "cosine"
    scanned: int = 0        # rows actually used for the covariance/reservoir

    @property
    def eigs(self) -> np.ndarray:
        return metrics.eigs_from_cov(self.cov)

    @property
    def approximate(self) -> bool:
        return self.scanned < self.n


class _NumpyCov:
    """Streaming covariance/sum/norm accumulator — float32 grams, float64 totals."""

    def __init__(self):
        self.n = 0
        self.dim = None
        self.sumv = self.XtX = None
        self.ns = self.nsq = 0.0

    def update(self, Xc):
        m, d = Xc.shape
        if self.dim is None:
            self.dim = d
            self.sumv = np.zeros(d, np.float64)
            self.XtX = np.zeros((d, d), np.float64)
        elif d != self.dim:
            raise ValueError(f"inconsistent embedding dim: {d} != {self.dim}")
        self.sumv += Xc.sum(0, dtype=np.float64)
        self.XtX += (Xc.T @ Xc).astype(np.float64, copy=False)   # sgemm, then accumulate in f64
        nr = np.linalg.norm(Xc, axis=1)
        self.ns += float(nr.sum())
        self.nsq += float((nr * nr).sum())
        self.n += m

    def finalize(self):
        mean = self.sumv / self.n
        cov = (self.XtX - self.n * np.outer(mean, mean)) / max(1, self.n - 1)
        nm = self.ns / self.n
        nsd = float(np.sqrt(max(0.0, self.nsq / self.n - nm * nm)))
        return self.dim, mean, cov, nm, nsd


def _source_rows(source) -> Optional[int]:
    """Cheap true row count (parquet metadata / npy shape) so approx mode can still
    report the real corpus size. Returns None when it would require a full read."""
    if isinstance(source, np.ndarray):
        return len(source)
    p = Path(source)
    suf = p.suffix.lower()
    try:
        if suf in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            return pq.ParquetFile(p).metadata.num_rows
        if suf == ".npy":
            return int(np.load(p, mmap_mode="r").shape[0])
    except Exception:
        return None
    return None


def scan(source, *, sample: int = 20_000, seed: int = 0, embedding_col=None,
         id_col=None, label_col=None, metric: str = "cosine",
         batch_rows: int = 50_000, device: str = "cpu", approx: Optional[int] = None) -> Scan:
    if device and device != "cpu":
        from . import accel
        acc = accel.TorchCov(accel.resolve_device(device))
    else:
        acc = _NumpyCov()

    rng = np.random.default_rng(seed)
    res = res_ids = res_lab = None
    scanned = 0

    for ch in iter_chunks(source, embedding_col=embedding_col, id_col=id_col,
                          label_col=label_col, batch_rows=batch_rows):
        Xc = np.ascontiguousarray(ch.X, dtype=np.float32)
        m, d = Xc.shape
        acc.update(Xc)

        if res is None:
            res = np.zeros((sample, d), dtype=np.float32)
            res_ids = np.empty(sample, dtype=object) if ch.ids is not None else None
            res_lab = np.empty(sample, dtype=object) if ch.labels is not None else None
        filled = min(scanned, sample)
        take = min(sample - filled, m)
        if take > 0:
            res[filled:filled + take] = Xc[:take]
            if res_ids is not None: res_ids[filled:filled + take] = ch.ids[:take]
            if res_lab is not None: res_lab[filled:filled + take] = ch.labels[:take]
        if m > take:
            rest = Xc[take:]
            t = scanned + take + np.arange(1, (m - take) + 1)
            j = rng.integers(0, t)
            sel = j < sample
            slots = j[sel]
            res[slots] = rest[sel]
            if res_ids is not None: res_ids[slots] = ch.ids[take:][sel]
            if res_lab is not None: res_lab[slots] = ch.labels[take:][sel]
        scanned += m
        if approx and scanned >= approx:
            break

    if scanned == 0:
        raise ValueError("empty source: no rows ingested")
    dim, mean, cov, norm_mean, norm_std = acc.finalize()
    keep = min(scanned, sample)
    es = EmbeddingSet(res[:keep],
                      ids=None if res_ids is None else res_ids[:keep],
                      labels=None if res_lab is None else res_lab[:keep],
                      metric=metric, source=str(source))
    total = _source_rows(source) or scanned
    return Scan(max(total, scanned), dim, cov, mean, norm_mean, norm_std,
                es, str(source), metric, scanned)
