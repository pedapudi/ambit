"""Projection to 2-D / 3-D for the visual layers. PCA is the default — fast,
deterministic, dependency-light (numpy SVD); UMAP is optional for non-linear
structure. Projection runs on the reservoir sample, never the full corpus."""

from __future__ import annotations

import numpy as np

from .types import EmbeddingSet


def pca(X: np.ndarray, n_components: int = 2):
    A = X - X.mean(0, keepdims=True)
    # economy SVD: reservoir is ~20k x d, so this is cheap
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    comps = Vt[:n_components]
    coords = A @ comps.T
    ev = (S[:n_components] ** 2) / max(1, A.shape[0] - 1)
    return coords, ev


def project(es: EmbeddingSet, n_components: int = 2, method: str = "pca", **kw) -> np.ndarray:
    X = es.X
    if method == "pca":
        return pca(X, n_components)[0]
    if method == "umap":
        import umap  # optional: pip install 'ambit[umap]'
        metric = "cosine" if es.metric == "cosine" else "euclidean"
        return umap.UMAP(n_components=n_components, metric=metric, **kw).fit_transform(X)
    raise ValueError(f"unknown projector {method!r}; use 'pca' or 'umap'")
