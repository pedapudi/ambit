"""Continuous occupancy — the pair-distance distribution read over all scales.

The theory is developed in docs/concepts/continuous-occupancy.md. The objects here
replace binned occupancy statistics with continuous, null-calibrated functionals of
the random-pair cosine sample ambit already collects:

- the **crowding curve** K: the fraction of pairs closer than each scale — an exact
  CDF with no bins, read against the analytic uniform-sphere null (Ripley's K on the
  sphere; the sphere has no boundary, so no edge correction exists to get wrong);
- the **occupancy discrepancy** (Stolarsky's invariance principle): the mean pairwise
  chord distance IS the all-caps-all-scales spherical-cap L2 discrepancy, reported as
  a z-score against the matched uniform null;
- an **anisotropy-conditioned null** (angular central Gaussian with the corpus's own
  covariance spectrum), so cone-shaped-but-benign corpora are not flagged as crowded;
- **confusion units**: under the Gaussian query channel q = x + sigma*g, a competitor
  at chord r out-scores the target with probability exactly Phi(-r/(2*sigma)); this
  converts the pair sample into expected retrieval collisions per entity and into the
  **resolution bandwidth** sigma* — the largest query noise the corpus tolerates
  before crowding exceeds a tolerance. Union-bound conservative under any competitor
  correlation.

Pure numpy. All samplers draw pair *cosines* directly (the pair cosine of uniform
vectors on S^{d-1} is 2*Beta((d-1)/2,(d-1)/2)-1), so no d-dimensional vectors are
materialized for the uniform null.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- small math
def _erf(x):
    """Vectorized erf (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7). numpy has no erf
    and scipy is optional, so ambit carries its own."""
    x = np.asarray(x, np.float64)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
               + t * (-1.453152027 + t * 1.061405429))))
    return s * (1.0 - poly * np.exp(-a * a))


def _phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(np.asarray(z, np.float64) / np.sqrt(2.0)))


def chords_from_cos(cos):
    """Chord distance ‖x−y‖ = √(2−2·cos) for unit vectors."""
    return np.sqrt(np.clip(2.0 - 2.0 * np.asarray(cos, np.float64), 0.0, None))


# ---------------------------------------------------------------- null samplers
def null_pair_cos(dim: int, n_pairs: int, seed: int = 0) -> np.ndarray:
    """Pair cosines of iid uniform points on S^{dim-1}, sampled directly via
    cos = 2·Beta((d−1)/2, (d−1)/2) − 1 — no vectors materialized."""
    rng = np.random.default_rng(seed)
    a = max((dim - 1) / 2.0, 0.5)
    return 2.0 * rng.beta(a, a, int(n_pairs)) - 1.0


def acg_pair_cos(eigs, n_pairs: int, seed: int = 0, m: int = 4096) -> np.ndarray:
    """Pair cosines under the **angular central Gaussian** with the given covariance
    spectrum — normalize(√λ ⊙ g). This is the anisotropy-conditioned reference: it
    reproduces the corpus's second-moment cone but none of its clustering, so
    (data − ACG) isolates crowding *beyond* the cone. Only the spectrum matters (the
    pair-cosine law is rotation invariant), so no eigenvectors are needed."""
    lam = np.clip(np.asarray(eigs, np.float64), 0.0, None)
    lam = lam[lam > 0]
    if lam.size == 0:
        return null_pair_cos(1, n_pairs, seed)
    rng = np.random.default_rng(seed)
    m = int(min(m, max(64, n_pairs)))
    X = rng.standard_normal((m, lam.size)) * np.sqrt(lam)[None, :]
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    i = rng.integers(0, m, n_pairs)
    j = rng.integers(0, m, n_pairs)
    keep = i != j
    return np.einsum("ij,ij->i", X[i[keep]], X[j[keep]])


# ---------------------------------------------------------------- the crowding curve
def exceedance(cos_sample, grid) -> np.ndarray:
    """K at each grid cosine: the fraction of sampled pairs with cos ≥ grid value —
    the empirical pair-closeness CDF, evaluated exactly (sort + searchsorted; no
    bins, no origin, no width)."""
    s = np.sort(np.asarray(cos_sample, np.float64))
    n = len(s)
    g = np.asarray(grid, np.float64)
    return (n - np.searchsorted(s, g, side="left")) / max(n, 1)


def null_envelope(dim: int, n_pairs: int, grid, reps: int = 19, seed: int = 0):
    """Pointwise max (and mean) of K over `reps` uniform-null replicates with the
    same pair-sample size — the graphical envelope the data curve is read against.
    (A pointwise band is a display device; the honest single p-value is a rank
    envelope, which this max approximates for the small-scale tail where the null
    is essentially zero.)"""
    ks = np.stack([exceedance(null_pair_cos(dim, n_pairs, seed + 7 * r), grid)
                   for r in range(reps)])
    return ks.max(0), ks.mean(0)


def liftoff_cos(cos_sample, dim: int, grid=None, reps: int = 19, seed: int = 0):
    """The largest pair-cosine at which the data curve exceeds the uniform-null
    envelope — the scale where crowding begins. None if the data never exceeds it."""
    cos_sample = np.asarray(cos_sample, np.float64)
    if grid is None:
        grid = np.linspace(0.999, max(float(cos_sample.mean()), 0.0), 160)
    env_max, _ = null_envelope(dim, len(cos_sample), grid, reps=reps, seed=seed)
    k = exceedance(cos_sample, grid)
    above = np.flatnonzero(k > env_max)
    return (float(grid[above[0]]), np.asarray(grid), k, env_max) if above.size \
        else (None, np.asarray(grid), k, env_max)


# ---------------------------------------------------------------- Stolarsky scalar
def stolarsky(cos_sample) -> float:
    """Mean pairwise chord distance. By Stolarsky's invariance principle (1973) this
    is, up to constants, the spherical-cap L2 discrepancy integrated over every cap
    center and radius — 'all bins at all scales' as one number. Larger = better
    spread; crowding shortens it."""
    return float(chords_from_cos(cos_sample).mean())


def stolarsky_z(cos_sample, dim: int, reps: int = 64, seed: int = 0):
    """(scalar, z): the mean-chord occupancy discrepancy and its z-score against the
    uniform null with a *matched* pair-sample size, so sampling noise is priced in.
    In high d the null is razor-thin (sd of the mean ~ 1e-5), so real crowding shows
    as a very large negative z."""
    n_pairs = len(np.asarray(cos_sample))
    s = stolarsky(cos_sample)
    means = np.array([chords_from_cos(null_pair_cos(dim, n_pairs, seed + 13 * r)).mean()
                      for r in range(reps)])
    sd = float(means.std()) or 1e-12
    return s, float((s - means.mean()) / sd)


# ---------------------------------------------------------------- confusion units
def pair_confusion(cos, sigma: float) -> np.ndarray:
    """Exact probability, under the Gaussian query channel q = x + sigma*g, that a
    competitor at this pair cosine out-scores the intended target: Phi(−‖x−y‖/(2σ)).
    Exact for any dimension (the score gap is exactly Gaussian)."""
    return _phi(-chords_from_cos(cos) / (2.0 * max(float(sigma), 1e-12)))


def expected_collisions(cos_sample, n: int, sigma: float) -> float:
    """C(σ): expected number of competitors that out-score the intended target for a
    typical entity, at query-noise scale σ. Union-bound conservative regardless of
    competitor correlations — a guarantee, not just an estimate."""
    return float((max(int(n), 2) - 1) * pair_confusion(cos_sample, sigma).mean())


def sigma_star(cos_sample, n: int, tol: float = 1.0,
               lo: float = 1e-3, hi: float = 3.0, iters: int = 60):
    """Resolution bandwidth σ*: the largest query-noise scale at which the expected
    collision count stays below `tol` (C is monotone increasing in σ; bisection).
    Returns hi if even σ=hi stays under tol, lo if the corpus is already over
    tolerance at σ=lo (i.e. effectively zero noise budget)."""
    cos_sample = np.asarray(cos_sample, np.float64)
    if expected_collisions(cos_sample, n, lo) >= tol:
        return float(lo)
    if expected_collisions(cos_sample, n, hi) <= tol:
        return float(hi)
    a, b = lo, hi
    for _ in range(iters):
        mid = np.sqrt(a * b)                     # bisect in log space
        if expected_collisions(cos_sample, n, mid) <= tol:
            a = mid
        else:
            b = mid
    return float(a)


def collision_counts(Xn, sigma: float, block: int = 2048) -> np.ndarray:
    """Per-entity expected collision count c_σ(x) = Σ_j Phi(−‖x−x_j‖/(2σ)) over the
    reservoir, blocked so the full Gram is never materialized. `Xn` must be
    L2-normalized. Names the entities at risk, with count (not score) semantics."""
    Xn = np.ascontiguousarray(Xn, np.float32)
    m = len(Xn)
    out = np.zeros(m, np.float64)
    Xt = np.ascontiguousarray(Xn.T)
    for s in range(0, m, block):
        e = min(s + block, m)
        G = (Xn[s:e] @ Xt).astype(np.float64)
        P = pair_confusion(np.clip(G, -1.0, 1.0), sigma)
        P[np.arange(e - s), s + np.arange(e - s)] = 0.0          # drop self
        out[s:e] = P.sum(1)
    return out
