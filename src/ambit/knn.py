"""k-nearest-neighbors over the reservoir — the structure behind the kNN-graph,
bridge, NN-margin and hubness figures. Exact by default (scikit-learn, or a numpy
brute fallback for small samples); the `ann` extra swaps in pynndescent so this
scales to a large reservoir. Everything downstream reads (idx, dist) and is
backend-agnostic.
"""

from __future__ import annotations

import numpy as np

from .types import EmbeddingSet


def knn(es: EmbeddingSet, k: int = 10, *, backend: str = "auto"):
    """Return (idx, dist): (m, k) neighbor indices and distances, self excluded.
    For cosine, dist = 1 - cosine_similarity."""
    cosine = es.metric == "cosine"
    U = es.normalize().X if cosine else es.X
    m = len(U)
    kk = min(k, m - 1)

    if backend in ("auto", "pynndescent"):
        try:
            from pynndescent import NNDescent
            index = NNDescent(U, metric="cosine" if cosine else "euclidean",
                              n_neighbors=kk + 1)
            idx, dist = index.neighbor_graph
            return _drop_self(idx, dist)
        except ImportError:
            if backend == "pynndescent":
                raise

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine" if cosine else "euclidean")
        nn.fit(U)
        dist, idx = nn.kneighbors(U)
        return _drop_self(idx, dist)
    except ImportError:
        pass

    if m > 8000:
        raise RuntimeError(
            f"exact kNN over {m} points needs scikit-learn or an ANN backend "
            "(pip install 'ambit[reduce]' or 'ambit[ann]')")
    return _brute(U, kk, cosine)


def _drop_self(idx, dist):
    out_i, out_d = [], []
    for r in range(len(idx)):
        keep = idx[r] != r
        out_i.append(idx[r][keep][: idx.shape[1] - 1])
        out_d.append(dist[r][keep][: dist.shape[1] - 1])
    return np.asarray(out_i), np.asarray(out_d)


def _brute(U, k, cosine):
    if cosine:
        S = U @ U.T
        np.fill_diagonal(S, -np.inf)
        idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
        rows = np.arange(len(U))[:, None]
        order = np.argsort(-S[rows, idx], axis=1)
        idx = idx[rows, order]
        return idx, 1.0 - S[rows, idx]
    d2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
    rows = np.arange(len(U))[:, None]
    order = np.argsort(d2[rows, idx], axis=1)
    idx = idx[rows, order]
    return idx, np.sqrt(d2[rows, idx])


def hubness(idx) -> np.ndarray:
    """k-occurrence: how often each point is *somebody's* neighbor (high skew = hubs)."""
    return np.bincount(idx.reshape(-1), minlength=len(idx))
