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

    if backend == "faiss":
        return faiss_knn(U, k)
    if backend == "brute":
        return _brute(U, kk, cosine)

    if backend in ("auto", "pynndescent"):
        try:
            from pynndescent import NNDescent
            idx, dist = NNDescent(U, metric="cosine" if cosine else "euclidean",
                                  n_neighbors=kk + 1).neighbor_graph
            return _drop_self(idx, dist)
        except ImportError:
            if backend == "pynndescent":
                raise

    if backend in ("auto", "sklearn"):
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine" if cosine else "euclidean").fit(U)
            dist, idx = nn.kneighbors(U)
            return _drop_self(idx, dist)
        except ImportError:
            if backend == "sklearn":
                raise

    if m > 8000:
        raise RuntimeError(
            f"exact kNN over {m} points needs scikit-learn or an ANN backend "
            "(pip install 'ambit[reduce]' or 'ambit[ann]')")
    return _brute(U, kk, cosine)


def faiss_knn(Xn, k: int):
    """FAISS kNN (GPU if faiss-gpu is present). Xn assumed L2-normalized; cosine via
    inner product. Returns (idx, dist=1-cos), self excluded."""
    import faiss
    X = np.ascontiguousarray(Xn, dtype=np.float32)
    n, d = X.shape
    index = faiss.IndexFlatIP(d)
    if hasattr(faiss, "StandardGpuResources"):
        try:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        except Exception:
            pass
    index.add(X)
    sims, ind = index.search(X, min(k, n - 1) + 1)
    return ind[:, 1:], (1.0 - sims[:, 1:]).astype(np.float32)


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


def topk_cosine(Xn, k: int, *, block: int = 2048) -> np.ndarray:
    """Top-k cosine-nearest neighbor **indices** per row (self excluded), blocked so the
    full m×m similarity is never materialized. `Xn` must be L2-normalized. Returns
    (m, k) int32. Used to compare two embeddings' neighborhoods item-by-item."""
    Xn = np.ascontiguousarray(Xn, np.float32)
    m = len(Xn)
    k = int(min(k, m - 1))
    idx = np.empty((m, max(k, 1)), np.int32)
    Xt = np.ascontiguousarray(Xn.T)
    for s in range(0, m, block):
        e = min(s + block, m)
        S = Xn[s:e] @ Xt
        S[np.arange(e - s), s + np.arange(e - s)] = -2.0          # drop self
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        idx[s:e, :k] = part[rows, np.argsort(-S[rows, part], axis=1)]
    return idx[:, :k]


def reciprocal_mask(idx) -> np.ndarray:
    """(m, k) bool: True where neighbor j of row i is *reciprocal* — i.e. i is also
    among j's listed neighbors. Vectorized via packed directed-edge keys (i·m + j)."""
    idx = np.asarray(idx)
    m, k = idx.shape
    r = np.repeat(np.arange(m, dtype=np.int64), k)
    c = idx.reshape(-1).astype(np.int64)
    valid = (c >= 0) & (c < m) & (c != r)
    fwd = np.sort((r * m + c)[valid])              # directed edges i->j present
    rev = c * m + r                                # the edge j->i we need to also exist
    return (np.isin(rev, fwd) & valid).reshape(m, k)


def reciprocal_filter(idx, dist):
    """Keep only reciprocal (mutual) neighbors, re-packed nearest-first, and pad the
    rest of each row with a gather-safe **self-loop sentinel** (idx = the row index,
    dist = +inf) so the graph stays a dense (m, k) array. Padding is identifiable as
    ``idx == row`` (equivalently ``dist == inf``); the hubness / purity / margin /
    sparsity readers skip it. Mutual-kNN suppresses hubs by construction (a hub that is
    in many lists but reciprocates few keeps only the few)."""
    idx = np.asarray(idx)
    dist = np.asarray(dist, float)
    m, k = idx.shape
    mask = reciprocal_mask(idx)
    oi = np.empty_like(idx)
    od = np.empty_like(dist)
    for r in range(m):
        keep = np.flatnonzero(mask[r])
        if keep.size:
            order = keep[np.argsort(dist[r, keep], kind="stable")]
            p = order.size
            oi[r, :p] = idx[r, order]
            od[r, :p] = dist[r, order]
        else:
            p = 0
        oi[r, p:] = r                              # self-loop sentinel (gather-safe)
        od[r, p:] = np.inf
    return oi, od
