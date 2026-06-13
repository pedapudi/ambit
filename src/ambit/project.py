"""Projection to 2-D / 3-D for the visual layers. PCA is the default — fast,
deterministic, dependency-light; `randomized=True` uses a randomized SVD that only
estimates the top components (much faster on a large reservoir, ~exact for 2-3
dims). UMAP is optional for non-linear structure. Projection runs on the reservoir
sample, never the full corpus.
"""

from __future__ import annotations

import numpy as np

from .types import EmbeddingSet


def pca(X: np.ndarray, n_components: int = 2, randomized: bool = False, seed: int = 0):
    A = X - X.mean(0, keepdims=True)
    if randomized:
        try:
            from sklearn.utils.extmath import randomized_svd
            _, S, Vt = randomized_svd(A, n_components=n_components, random_state=seed)
            comps = Vt[:n_components]
            return A @ comps.T, S
        except ImportError:
            pass
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    comps = Vt[:n_components]
    return A @ comps.T, (S[:n_components] ** 2) / max(1, A.shape[0] - 1)


def project(es: EmbeddingSet, n_components: int = 2, method: str = "pca",
            randomized: bool = False, **kw) -> np.ndarray:
    X = es.X
    if method == "pca":
        return pca(X, n_components, randomized=randomized)[0]
    if method == "umap":
        import umap  # optional: pip install 'ambit[umap]'
        metric = "cosine" if es.metric == "cosine" else "euclidean"
        return umap.UMAP(n_components=n_components, metric=metric, **kw).fit_transform(X)
    raise ValueError(f"unknown projector {method!r}; use 'pca' or 'umap'")
