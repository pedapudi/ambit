"""Label-aware separability — does the partition occupy distinct regions of the
space? The partition is either the **provided** labels or, when none are given,
ambit's own **clusters** of the geometry (`cluster.cluster`, via `Ctx.labels`).

It computes the per-label geometry — the centroid-cosine matrix and kNN purity that
carry the interpretation — and, for *discovered* structure, the readouts that only make
sense there: how **stable** the clusters are (bootstrap ARI) and how many **modes** the
graph spectrum implies (Laplacian eigengap), so the panel is honest about an
unsupervised partition.

Pure numpy except `cluster_stability` (sklearn's ARI + k-means, guarded and optional).
Everything runs on the reservoir (separability is a structural property; no need for
the full corpus). The cosine→geometry read is *geometric*, not semantic: tidy clusters
can be meaningless and true categories can live on curved manifolds — which is why the
unsupervised branch leads with stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metrics import _unit


def _groups(y) -> list:
    """Sorted unique group labels, excluding the integer -1 HDBSCAN noise marker."""
    vals = set(np.asarray(y).tolist())
    vals.discard(-1)
    return sorted(vals)


def centroid_cosine_matrix(X, y, groups) -> np.ndarray:
    """g×g cosine matrix between per-group mean unit vectors (the "which subdomains
    collapse together" matrix). Off-diagonal near 1 = two groups share a direction."""
    U = _unit(np.asarray(X, float))
    y = np.asarray(y)
    cents = np.vstack([U[y == s].mean(0) for s in groups])
    cents = _unit(cents)
    return cents @ cents.T


def knn_label_purity(knn_idx, y):
    """Per point, the fraction of its kNN that share its label. Returns
    (overall, per_point) — does a doc's neighborhood agree with its label? Self-loop
    padding (idx == row, from a reciprocal/mutual graph) is excluded; a point with no
    reciprocal neighbor gets NaN and is dropped from the averages."""
    y = np.asarray(y)
    idx = np.asarray(knn_idx)
    rows = np.arange(len(idx))[:, None]
    real = idx != rows                                  # drop self-loop padding (mutual-kNN)
    match = (y[idx] == y[:, None]) & real
    cnt = real.sum(1)
    per_point = np.where(cnt > 0, match.sum(1) / np.maximum(cnt, 1), np.nan)
    overall = float(np.nanmean(per_point)) if np.isfinite(per_point).any() else 0.0
    return overall, per_point


def fisher_ratio(X, y, groups) -> float:
    """tr(S_b)/tr(S_w): between- over within-group scatter (higher = more separable),
    in trace form so it is O(n·d) with no d×d scatter matrices."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    mu = X.mean(0)
    sw = sb = 0.0
    for s in groups:
        Xs = X[y == s]
        if len(Xs) == 0:
            continue
        mus = Xs.mean(0)
        sw += float(((Xs - mus) ** 2).sum())
        sb += float(len(Xs) * ((mus - mu) ** 2).sum())
    return float(sb / sw) if sw > 0 else 0.0


def silhouette_cosine(X, y, groups, *, sample: int = 2000, seed: int = 0) -> float:
    """Mean silhouette under cosine distance on a capped subsample (O(s²·d)), pure
    numpy. +1 = tight, well-separated groups; ≈0 = overlapping; <0 = mixed."""
    y = np.asarray(y)
    if len(groups) < 2:
        return 0.0
    U = _unit(np.asarray(X, float))
    rng = np.random.default_rng(seed)
    if len(U) > sample:
        idx = rng.choice(len(U), sample, replace=False)
        U, y = U[idx], y[idx]
    keep = np.isin(y, groups)                           # drop noise points
    U, y = U[keep], y[keep]
    s = len(U)
    if s < 3:
        return 0.0
    D = 1.0 - U @ U.T                                   # cosine distance (s×s)
    np.fill_diagonal(D, 0.0)
    G = np.vstack([(y == g) for g in groups]).astype(float)   # (g, s)
    counts = G.sum(1)                                   # (g,)
    gi = {g: k for k, g in enumerate(groups)}
    own = np.array([gi[v] for v in y])                  # group index per point
    sums = D @ G.T                                      # (s, g) summed distances
    a = sums[np.arange(s), own] / np.maximum(counts[own] - 1, 1)   # exclude self
    other = sums / np.maximum(counts, 1)
    other[np.arange(s), own] = np.inf
    b = other.min(1)
    denom = np.maximum(a, b)
    sil = np.where(denom > 0, (b - a) / denom, 0.0)
    return float(sil.mean())


def cluster_stability(X, y, groups, *, n_boot: int = 8, frac: float = 0.8,
                      seed: int = 0) -> Optional[float]:
    """Mean adjusted-Rand index between the partition and k-means re-clusterings on
    bootstrap subsamples — does the discovered structure reproduce? High ≈ trust the
    heatmap; low ⇒ the clusters are an artifact of one run. None if sklearn is absent."""
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return None
    g = len(groups)
    if g < 2:
        return None
    X = np.ascontiguousarray(X, np.float32)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), max(g + 1, int(frac * len(X))), replace=False)
        lab = MiniBatchKMeans(n_clusters=g, n_init=3, batch_size=max(1024, len(idx) // 8),
                              random_state=int(rng.integers(1 << 30))).fit_predict(X[idx])
        aris.append(adjusted_rand_score(y[idx], lab))
    return float(np.mean(aris))


def laplacian_eigengap(X, *, k: int = 10, sample: int = 1200, kmax: int = 12,
                       seed: int = 0):
    """Estimate the number of well-separated modes from the spectrum of the normalized
    graph Laplacian on a subsample: the index of the largest gap among the smallest
    eigenvalues. Pure numpy (dense eigvalsh on the subsample). Returns (n_modes, evals)."""
    U = _unit(np.asarray(X, float))
    rng = np.random.default_rng(seed)
    if len(U) > sample:
        U = U[rng.choice(len(U), sample, replace=False)]
    n = len(U)
    if n < 4:
        return 1, np.zeros(1)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, n - 1)
    nbr = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), kk), nbr.ravel()] = 1.0
    W = np.maximum(W, W.T)                              # symmetrize the kNN graph
    d = W.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    L = np.eye(n) - (dinv[:, None] * W * dinv[None, :])
    ev = np.sort(np.linalg.eigvalsh(L))[: kmax + 1]
    gaps = np.diff(ev)
    n_modes = int(np.argmax(gaps) + 1) if gaps.size else 1
    return n_modes, ev


@dataclass
class Separability:
    """Everything the RES 05b panel draws. `supervised` distinguishes a provided
    partition from ambit's discovered clusters (which carry stability + n_modes)."""
    groups: list
    counts: list
    centroids_cos: np.ndarray            # (g, g)
    purity_overall: Optional[float]
    purity_per_group: Optional[np.ndarray]
    silhouette: float
    fisher: float
    stability: Optional[float]           # unsupervised only
    n_modes: Optional[int]               # unsupervised only
    supervised: bool


def compute(X, y, knn_idx=None, *, supervised: bool = True, seed: int = 0) -> Optional[Separability]:
    """Build the separability panel for a partition `y` over the reservoir `X`.
    Returns None when there are fewer than two groups (the figure then degrades)."""
    y = np.asarray(y)
    groups = _groups(y)
    if len(groups) < 2:
        return None
    X = np.asarray(X, float)
    counts = [int((y == s).sum()) for s in groups]
    C = centroid_cosine_matrix(X, y, groups)

    purity_overall = purity_pg = None
    if knn_idx is not None:
        _, pp = knn_label_purity(knn_idx, y)
        mask = np.isin(y, groups) & np.isfinite(pp)
        purity_overall = float(pp[mask].mean()) if mask.any() else None
        purity_pg = np.array([
            float(np.nanmean(pp[y == s])) if np.isfinite(pp[y == s]).any() else 0.0
            for s in groups])

    sil = silhouette_cosine(X, y, groups, seed=seed)
    fish = fisher_ratio(X, y, groups)

    stability = n_modes = None
    if not supervised:
        stability = cluster_stability(X, y, groups, seed=seed)
        n_modes, _ = laplacian_eigengap(X, seed=seed)

    return Separability(groups, counts, C, purity_overall, purity_pg,
                        sil, fish, stability, n_modes, supervised)
