"""Two-embedding comparison — how much, and *where*, two embeddings of the **same
items** differ. The headline use is one dataset embedded two ways (two encoders or
configurations): does the representation actually move, by how much, and where?

The scalars here are pure numpy:

  - **linear CKA** (Kornblith et al. 2019) — similarity in [0,1], invariant to rotation,
    isotropic scaling and neuron permutation; the exact, scalable headline (streams,
    O(d²) memory, and `d_A` may differ from `d_B`).
  - **RBF CKA** — the kernel form (nonlinear structure), O(n²), sampled.
  - **MMD²** and **energy distance** — distributional "did the cloud shape move"; both
    need cross-set distances, so they require `d_A == d_B`.
  - **Procrustes disparity** — best rigid-alignment residual; requires `d_A == d_B`.

The alignment of the two sets is by **stable id**, never row order — see
`CmpCtx` / `build_cmp`. When the dimensions differ (e.g. a Matryoshka-truncated set vs
full), only the two CKA variants are defined; the dimension-coupled metrics return None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------- scalar metrics
def _center_cols(X: np.ndarray) -> np.ndarray:
    return X - X.mean(0, keepdims=True)


def linear_cka(X, Y, *, batch_rows: int = 50_000) -> float:
    """Linear CKA in [0,1] — feature-space form ‖Y_cᵀX_c‖_F² / (‖X_cᵀX_c‖_F·‖Y_cᵀY_c‖_F).
    O(n·(d_x²+d_y²)) time, O(d²) memory; streams over rows so it runs on the full corpus,
    and `d_x` may differ from `d_y` (the cross term is d_y×d_x). 1 = identical up to
    rotation/scale/permutation; near 0 = unrelated."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    mx = X.mean(0)
    my = Y.mean(0)
    Sxy = Sxx = Syy = 0.0
    for s in range(0, n, batch_rows):
        a = X[s:s + batch_rows] - mx
        b = Y[s:s + batch_rows] - my
        Sxy = Sxy + b.T @ a                 # (d_y, d_x)
        Sxx = Sxx + a.T @ a                 # (d_x, d_x)
        Syy = Syy + b.T @ b                 # (d_y, d_y)
    fro = lambda M: float(np.sqrt(float((np.asarray(M) ** 2).sum())))
    return fro(Sxy) ** 2 / (fro(Sxx) * fro(Syy) + 1e-12)


def _rbf_gram(X, sigma: Optional[float] = None) -> np.ndarray:
    X = np.asarray(X, float)
    sq = np.einsum("ij,ij->i", X, X)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    if sigma is None:
        n = len(X)
        iu = np.triu_indices(n, 1)
        med = float(np.median(np.sqrt(D2[iu]))) if iu[0].size else 1.0
        sigma = med or 1.0
    return np.exp(-D2 / (2.0 * sigma * sigma))


def _hsic(K, L) -> float:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H)) / float((n - 1) ** 2)


def rbf_cka(X, Y) -> float:
    """RBF (kernel) CKA in [0,1] — captures nonlinear structure. O(n²): call on a
    sample (≤ cmp_kernel_sample). σ = median pairwise distance, per set."""
    Kx = _rbf_gram(X)
    Ky = _rbf_gram(Y)
    return _hsic(Kx, Ky) / (np.sqrt(_hsic(Kx, Kx) * _hsic(Ky, Ky)) + 1e-12)


def _median_sigma(X, Y) -> float:
    Z = np.vstack([np.asarray(X, float), np.asarray(Y, float)])
    sq = np.einsum("ij,ij->i", Z, Z)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(D2, 0.0, out=D2)
    iu = np.triu_indices(len(Z), 1)
    med = float(np.median(np.sqrt(D2[iu]))) if iu[0].size else 1.0
    return med or 1.0


def mmd2_rbf(X, Y, sigma: Optional[float] = None) -> float:
    """Unbiased MMD² with an RBF kernel — a distributional distance (did the cloud
    shape move). O((n+m)²): sampled. Requires d_x == d_y (cross kernel needs ‖x−y‖)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if sigma is None:
        sigma = _median_sigma(X, Y)
    g = 1.0 / (2.0 * sigma * sigma)

    def k(A, B):
        sa = np.einsum("ij,ij->i", A, A)
        sb = np.einsum("ij,ij->i", B, B)
        D2 = sa[:, None] + sb[None, :] - 2.0 * (A @ B.T)
        np.maximum(D2, 0.0, out=D2)
        return np.exp(-g * D2)

    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(sxx + syy - 2.0 * Kxy.mean())


def energy_distance(X, Y) -> float:
    """Energy distance E² = 2·E‖X−Y‖ − E‖X−X′‖ − E‖Y−Y′‖ (≥0). O((n+m)²): sampled.
    Requires d_x == d_y."""
    def mean_dist(A, B):
        sa = np.einsum("ij,ij->i", A, A)
        sb = np.einsum("ij,ij->i", B, B)
        D2 = sa[:, None] + sb[None, :] - 2.0 * (A @ B.T)
        np.maximum(D2, 0.0, out=D2)
        return float(np.sqrt(D2).mean())

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    return float(max(0.0, 2.0 * mean_dist(X, Y) - mean_dist(X, X) - mean_dist(Y, Y)))


def procrustes_disparity(X, Y) -> float:
    """Orthogonal Procrustes residual ‖Y_c − X_c R‖_F² / ‖Y_c‖_F², R = UVᵀ from
    X_cᵀY_c = UΣVᵀ. 0 = identical up to rotation. Requires d_x == d_y."""
    Xc = _center_cols(np.asarray(X, float))
    Yc = _center_cols(np.asarray(Y, float))
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt
    return float(((Yc - Xc @ R) ** 2).sum() / (float((Yc ** 2).sum()) + 1e-12))


def neighbor_overlap(idx_a, idx_b) -> np.ndarray:
    """Per item, the fraction of its top-k neighbors **shared** between two graphs of the
    *same items in the same order* (set overlap / neighborhood retention). 1 = identical
    neighborhood, 0 = fully reshuffled. This is the local, retrieval-relevant counterpart
    to CKA: it compares neighbor *identities*, so it is **dimension-agnostic** (works even
    when d_A != d_B, unlike the drift field) and it sees the local reshuffling a global
    second-moment statistic like CKA can miss."""
    a = np.asarray(idx_a)
    b = np.asarray(idx_b)
    k = a.shape[1]
    out = np.empty(len(a))
    for i in range(len(a)):
        out[i] = len(set(a[i].tolist()) & set(b[i].tolist())) / k
    return out


# --------------------------------------------------------------- alignment + ctx
def resolve_rows_by_id(ids, wanted):
    """Map `wanted` ids to row positions in `ids` (O(n+m) via a dict; first occurrence
    wins). Returns (rows, found): `found` is a bool mask over `wanted`, `rows[k]` the row
    of `wanted[k]` in `ids` (or -1). Keyed by str() so an int id in one set matches a
    str id in the other."""
    pos = {}
    for r, v in enumerate(np.asarray(ids).tolist()):
        pos.setdefault(str(v), r)
    rows = np.array([pos.get(str(v), -1) for v in np.asarray(wanted).tolist()], dtype=int)
    return rows, rows >= 0


@dataclass
class CmpCtx:
    label_a: str
    label_b: str
    ids: np.ndarray                  # (m',) aligned ids
    A: np.ndarray                    # (m', d_a) aligned, L2-normalized reservoir of A
    B: np.ndarray                    # (m', d_b) aligned, L2-normalized rows of B (same items)
    xyA: np.ndarray                  # (m', 2) A in A's PCA frame
    xyB_in_A: Optional[np.ndarray]   # (m', 2) B through A's PCA basis (same-dim only)
    same_dim: bool
    cka_linear: float                # exact, scalable headline (any dims)
    cka_rbf: Optional[float]
    mmd2: Optional[float]            # None when d_a != d_b
    energy: Optional[float]          # None when d_a != d_b
    procrustes_disp: Optional[float] # None when d_a != d_b
    drift: Optional[np.ndarray]      # (m',) ‖xyB_in_A − xyA‖ ; None when d_a != d_b
    drift_cos: Optional[np.ndarray]  # (m',) cos(A_i, B_i) native ; None when d_a != d_b
    uniformity_series: list          # [(label_a, U_A), (label_b, U_B)] for RES 08
    nbr_overlap: np.ndarray          # (m',) per-item top-k neighbor retention (A vs B)
    nbr_overlap_mean: float          # mean retention at the primary k (the local headline)
    nbr_overlap_by_k: dict           # {k: mean retention} across scales
    nbr_k: int                       # the primary k


def _unit_rows(X):
    X = np.asarray(X, float)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def build_cmp(ctx, cfg) -> "CmpCtx":
    """Align a second embedding set (`cfg.compare`) to A's reservoir by **stable id** and
    compute the comparison. Raises ValueError on the id preconditions — the user asked to
    compare, so failing loudly beats a silently meaningless number (two arange-id sets
    would otherwise 'intersect' on every row and pair unrelated items)."""
    from .metrics import uniformity
    from .project import pca_fit, pca_transform
    from .source import iter_chunks

    if not getattr(ctx.es, "ids_provided", False):
        raise ValueError(
            "--compare needs stable ids on the primary set: pass --id-col (or use a format "
            "carrying an id column). The reservoir fell back to row indices, which cannot be "
            "matched across two independently sampled sets.")
    id_col = cfg.compare_id_col or cfg.id_col

    # Stream the second set and collect only the rows whose id matches A's reservoir, so
    # B is never fully resident (a 1M-row compare set keeps ~20k rows here) and a sharded
    # directory / glob works exactly like a single file. We compare on the reservoir.
    a_ids = np.asarray(ctx.es.ids)
    want = {str(v): i for i, v in enumerate(a_ids.tolist())}     # reservoir id -> reservoir row
    collected = {}                                              # reservoir row -> B vector
    saw_ids = False
    for ch in iter_chunks(cfg.compare, embedding_col=cfg.embedding_col,
                          id_col=id_col, batch_rows=cfg.batch_rows):
        if ch.ids is None:
            break
        saw_ids = True
        for k, vid in enumerate(ch.ids.tolist()):
            r = want.get(str(vid))
            if r is not None and r not in collected:
                collected[r] = ch.X[k]
        if len(collected) >= len(want):
            break
    if not saw_ids:
        raise ValueError(f"the compare set {cfg.compare!r} has no id column to align on; "
                         "pass --compare-id-col (or --id-col).")
    keep = np.array(sorted(collected), dtype=int)
    if keep.size == 0:
        raise ValueError(
            f"no overlapping ids between the two sets — nothing to compare (checked "
            f"{len(a_ids)} reservoir ids against the compare set).")

    A = _unit_rows(ctx.es.X[keep])
    B = _unit_rows(np.asarray([collected[r] for r in keep], dtype=np.float32))
    ids = a_ids[keep]
    same_dim = (A.shape[1] == B.shape[1])

    # A's PCA frame — always linear PCA for the drift frame (independent of ctx.xy's projector)
    xyA_full, mean_a, comps_a, _ = pca_fit(ctx.es.X, 2)
    xyA = xyA_full[keep]
    xyB_in_A = pca_transform(B, mean_a, comps_a) if same_dim else None

    cka_linear = linear_cka(A, B, batch_rows=cfg.batch_rows)
    ks = int(min(cfg.cmp_kernel_sample, len(A)))
    if ks >= 8:
        cka_rbf = rbf_cka(A[:ks], B[:ks])
        mmd2 = mmd2_rbf(A[:ks], B[:ks]) if same_dim else None
        energy = energy_distance(A[:ks], B[:ks]) if same_dim else None
    else:
        cka_rbf = mmd2 = energy = None
    procrustes_disp = procrustes_disparity(A, B) if same_dim else None

    if same_dim:
        drift = np.linalg.norm(xyB_in_A - xyA, axis=1)
        drift_cos = np.einsum("ij,ij->i", A, B)
    else:
        drift = drift_cos = None

    # local, dimension-agnostic comparison: how much each item's neighborhood reshuffled
    # (the retrieval-relevant signal CKA can miss). Build each set's own kNN and compare.
    from . import knn as _knn
    K = int(min(50, len(A) - 1))
    if K >= 1:
        ia = _knn.topk_cosine(A, K)
        ib = _knn.topk_cosine(B, K)
        nbr_k = int(min(cfg.k, K))
        scales = sorted({nbr_k, K})
        nbr_overlap_by_k = {k: float(neighbor_overlap(ia[:, :k], ib[:, :k]).mean()) for k in scales}
        nbr_overlap = neighbor_overlap(ia[:, :nbr_k], ib[:, :nbr_k])
        nbr_overlap_mean = float(nbr_overlap.mean())
    else:
        nbr_overlap = np.zeros(len(A))
        nbr_overlap_mean = 1.0
        nbr_overlap_by_k = {}
        nbr_k = 0

    series = [("A", uniformity(A, normalized=True)),
              (cfg.compare_label, uniformity(B, normalized=True))]

    return CmpCtx(label_a="A", label_b=cfg.compare_label, ids=ids, A=A, B=B, xyA=xyA,
                  xyB_in_A=xyB_in_A, same_dim=same_dim, cka_linear=cka_linear,
                  cka_rbf=cka_rbf, mmd2=mmd2, energy=energy,
                  procrustes_disp=procrustes_disp, drift=drift, drift_cos=drift_cos,
                  uniformity_series=series, nbr_overlap=nbr_overlap,
                  nbr_overlap_mean=nbr_overlap_mean, nbr_overlap_by_k=nbr_overlap_by_k,
                  nbr_k=nbr_k)
