"""Native-space resolution / isotropy diagnostics — pure numpy, no heavy deps.

These run on the ORIGINAL high-dimensional vectors (not a 2-D projection); they
are the scalar backbone of the "resolution / isotropy" facet. See
docs/concepts/anisotropy-and-resolution.md for what each one means.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _unit(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def random_pair_cosine(X, n_pairs: int = 200_000, seed: int = 0, normalized: bool = False):
    """Sample of off-diagonal cosine similarities — the anisotropy fingerprint.
    Mean ~ 0 is isotropic (high resolution); mass shifted toward 1 is crowded."""
    n = len(X)
    U = X if normalized else _unit(X)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    return np.einsum("ij,ij->i", U[i[keep]], U[j[keep]]).astype(np.float64)


def cov_eigs(X) -> np.ndarray:
    """Descending non-negative eigenvalues of the centered covariance (computed via
    the d×d scatter, so it scales in n)."""
    A = X - X.mean(0, keepdims=True)
    return eigs_from_cov((A.T @ A) / max(1, A.shape[0] - 1))


def eigs_from_cov(cov: np.ndarray) -> np.ndarray:
    """Descending non-negative eigenvalues of a precomputed covariance matrix —
    lets the streaming scan compute exact rank metrics over the whole corpus."""
    w = np.linalg.eigvalsh(cov)
    return np.sort(np.clip(w, 0.0, None))[::-1]


def effective_rank(eigs: np.ndarray) -> float:
    """exp(entropy of normalized singular values) — continuous effective dimensionality."""
    s = np.sqrt(eigs[eigs > 0])
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigs: np.ndarray) -> float:
    e = eigs[eigs > 0]
    return float(e.sum() ** 2 / (e ** 2).sum())


def dims_for_variance(eigs: np.ndarray, frac: float = 0.9) -> int:
    c = np.cumsum(eigs) / eigs.sum()
    return int(np.searchsorted(c, frac) + 1)


def isotropy_ref(dim: int) -> float:
    """Std of random-pair cosine for iid points on the unit d-sphere (~N(0, 1/d))."""
    return 1.0 / np.sqrt(dim)


def isoscore_parts(eigs: np.ndarray):
    """IsoScore (Rudman, Gillman, Rush & Eickhoff, 2022) from the covariance
    eigenvalues λ — i.e. the variance along each principal axis. Returns
    (score, defect δ, n_iso, dim) so callers can show the intermediate terms.

      v        = √d · λ / ‖λ‖₂            (normalize: isotropic ⇒ v = all-ones)
      δ        = ‖v − 1‖₂ / √(2(d − √d))   (isotropy defect, 0=isotropic … 1=one axis)
      n_iso    = d − δ²(d − √d)            (dimensions isotropically used, √d … d)
      IsoScore = (n_iso² − d) / (d² − d)   (0 = degenerate line … 1 = perfect sphere)
    """
    lam = np.clip(np.asarray(eigs, float), 0.0, None)
    d = int(lam.size)
    if d <= 1:
        return float(d), 0.0, float(d), d
    nrm = float(np.linalg.norm(lam))
    if nrm == 0.0:
        return 0.0, 1.0, float(np.sqrt(d)), d
    sd = np.sqrt(d)
    v = sd * lam / nrm                                   # ‖v‖₂ = √d; isotropic ⇒ all-ones
    defect = float(np.linalg.norm(v - 1.0) / np.sqrt(2.0 * (d - sd)))
    defect = min(max(defect, 0.0), 1.0)
    n_iso = d - defect ** 2 * (d - sd)
    score = float((n_iso ** 2 - d) / (d ** 2 - d))
    return min(max(score, 0.0), 1.0), defect, float(n_iso), d


def isoscore(eigs: np.ndarray) -> float:
    """IsoScore in [0,1]: how uniformly variance fills the space (1 = isotropic)."""
    return isoscore_parts(eigs)[0]


def isoscore_star(eigs: np.ndarray, n_samples: int, zeta: Optional[float] = None) -> float:
    """Small-sample (n<d) IsoScore in the spirit of IsoScore* (Rudman & Eickhoff,
    2024). A cloud of n points spans at most n-1 dimensions, so plain IsoScore counts
    the d-(n-1) forced-zero directions as anisotropy and flags every small group as a
    cone. The paper's fix is RDA *shrinkage* of the covariance toward an isotropic
    reference, with the ambient dimension d held fixed (not truncated):

        Σ_ζ = (1-ζ)·Σ_X + ζ·(tr Σ_X / d)·I        (ζ ∈ [0,1); ζ=0 ⇒ plain IsoScore)

    The target is a scaled identity, so it shares eigenvectors with Σ_X and the blend
    is exact in eigenvalue space: λ_ζ = (1-ζ)·λ + ζ·mean(λ). Filling the forced-zero
    floor lets a genuinely round small cloud read high, while a pocket that has truly
    collapsed onto a low-dim subspace keeps its dominant axes and still reads low. ζ
    defaults to the rank-deficiency fraction 1-(n-1)/d — zero once n-1 ≥ d (the regime
    where IsoScore is already reliable), growing as the sample starves the dimensions.
    Below n ≈ d the value is a coarse round/collapsed indicator, not an exact isotropy.
    """
    lam = np.clip(np.asarray(eigs, float), 0.0, None)
    d = int(lam.size)
    total = float(lam.sum())
    if d <= 1 or total <= 0.0:
        return 1.0
    if zeta is None:
        zeta = float(np.clip(1.0 - min(max(n_samples - 1, 0), d) / d, 0.0, 0.95))
    if zeta <= 0.0:
        return isoscore(lam)
    lam_z = (1.0 - zeta) * lam + zeta * (total / d)
    return isoscore(lam_z)


def hubness_skew(knn_idx: np.ndarray) -> float:
    """Skewness of the k-occurrence distribution — how often each point is *somebody's*
    neighbor. High positive skew = a few hubs dominate retrieval (Radovanović et al.).
    Self-references are dropped, so a reciprocal (mutual) graph's self-loop padding does
    not pollute the count."""
    idx = np.asarray(knn_idx)
    rows = np.arange(len(idx))[:, None]
    flat = idx[idx != rows]                              # drop self-loop padding (mutual-kNN)
    occ = np.bincount(flat.reshape(-1), minlength=len(idx)).astype(np.float64)
    m = occ.mean()
    sd = occ.std() or 1.0
    return float(((occ - m) ** 3).mean() / sd ** 3)


def uniformity_from_cos(cos: np.ndarray, *, t: float = 2.0) -> float:
    """Wang–Isola (2020) uniformity from a random-pair cosine sample. For unit vectors
    ‖x−y‖² = 2 − 2·cos, so U = log E exp(−t·‖x−y‖²) is a function of the cosines ambit
    already samples — more negative = more uniformly spread on the sphere (better
    occupancy). Reuses `ctx.cos`, so the header scalar costs nothing extra."""
    sq = 2.0 - 2.0 * np.asarray(cos, dtype=np.float64)
    return float(np.log(np.mean(np.exp(-t * sq))))


def uniformity(X, *, t: float = 2.0, n_pairs: int = 200_000, seed: int = 0,
               normalized: bool = False) -> float:
    """Uniformity (Wang–Isola) over `n_pairs` random pairs of `X`. Mirrors the
    `random_pair_cosine` sampling, so it is O(n_pairs) and trivial on 1M rows."""
    U = X if normalized else _unit(X)
    rng = np.random.default_rng(seed)
    n = len(U)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    cos = np.einsum("ij,ij->i", U[i[keep]], U[j[keep]])
    return uniformity_from_cos(cos, t=t)


def uniformity_ref(dim: int, *, t: float = 2.0, n: int = 4000, seed: int = 0) -> float:
    """Uniformity of an isotropic cloud (iid uniform on the unit `dim`-sphere) — the
    reference the dataset's uniformity is read against (cf. `isotropy_ref` for cosine)."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, dim)).astype(np.float64)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    return uniformity(R, t=t, n_pairs=min(200_000, n * n), seed=seed, normalized=True)
