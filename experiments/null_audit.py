#!/usr/bin/env python3
"""Calibration audit of the anisotropy-conditioned reference.

Three questions, per the review that prompted this audit:

1. TRACKING --- which reference reproduces which geometry? Compares the
   centered-spectrum ACG, the uncentered-moment ACG, and the mean-aware
   Gaussian fit (`conditioned_pair_cos`) across mean cones, zero-mean
   ellipses, a two-cone mixture, and cone-plus-pocket corpora.

2. CALIBRATION --- the fit is a composite null: mu and Sigma are estimated
   from the corpus under test. A tail test against the fitted reference is
   calibrated only if each null replicate REFITS (mu, Sigma) before its
   reference is drawn (parametric bootstrap). Measures the false-positive
   rate of the refitted test on genuinely cone-shaped null corpora.

3. POWER / ABSORPTION --- a fitted reference could absorb a large planted
   pocket into its own mean and covariance and stop detecting it. Measures
   detection rate of the refitted test as pocket mass grows, and how far
   the fitted reference's tail rises with it.

Run:  python experiments/null_audit.py            (~2-4 min on CPU)
"""
import sys, os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ambit import occupancy as occ  # noqa: E402


# ------------------------------------------------------------ corpus builders
def cone(n, d, strength, rng):
    mu = rng.standard_normal(d)
    mu /= np.linalg.norm(mu)
    X = strength * np.sqrt(d) * mu + rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def ellipse(n, d, top, scale, rng):
    s = np.ones(d)
    s[:top] = scale
    X = rng.standard_normal((n, d)) * s
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def mixture(n, d, strength, rng):
    a = cone(n // 2, d, strength, rng)
    b = cone(n - n // 2, d, strength, rng)
    return np.vstack([a, b])


def add_pocket(X, frac, spread, rng):
    d = X.shape[1]
    k = max(2, int(frac * len(X)))
    c = rng.standard_normal(d)
    c /= np.linalg.norm(c)
    P = c + spread * rng.standard_normal((k, d))
    Y = X.copy()
    Y[-k:] = P / np.linalg.norm(P, axis=1, keepdims=True)
    return Y


# ------------------------------------------------------------ shared pieces
def pair_cos(X, P, rng):
    i = rng.integers(0, len(X), P)
    j = rng.integers(0, len(X), P)
    k = i != j
    return np.einsum("ij,ij->i", X[i[k]], X[j[k]])


def fit(X):
    mean = X.mean(0)
    cov = (X.T @ X - len(X) * np.outer(mean, mean)) / (len(X) - 1)
    return mean, cov


def K(cos, grid):
    return occ.exceedance(cos, grid)


# ------------------------------------------------------------ 1. tracking
def tracking(n=4000, d=128, P=100_000, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.05, 0.95, 40)
    cases = {
        "cone weak (cos~0.3)":   cone(n, d, 0.316, rng),
        "cone strong (cos~0.5)": cone(n, d, 0.577, rng),
        "cone extreme (cos~0.7)": cone(n, d, 1.0, rng),
        "ellipse mild":          ellipse(n, d, 8, 3.0, rng),
        "ellipse strong":        ellipse(n, d, 8, 6.0, rng),
        "two-cone mixture":      mixture(n, d, 0.577, rng),
    }
    print("== 1. tracking: max |K_ref - K_data| over the grid (smaller = better)")
    print(f"{'case':24s} {'ACG-centered':>13} {'ACG-uncentered':>15} {'Gaussian fit':>13}")
    for name, X in cases.items():
        data = K(pair_cos(X, P, rng), grid)
        mean, cov = fit(X)
        c_ref = K(occ.acg_pair_cos(np.linalg.eigvalsh(cov), P, seed=1), grid)
        mom = (X.T @ X) / len(X)
        u_ref = K(occ.acg_pair_cos(np.linalg.eigvalsh(mom), P, seed=1), grid)
        g_ref = K(occ.conditioned_pair_cos(mean, cov, P, seed=1), grid)
        row = [np.max(np.abs(r - data)) for r in (c_ref, u_ref, g_ref)]
        print(f"{name:24s} {row[0]:13.4f} {row[1]:15.4f} {row[2]:13.4f}")
    print()


# --------------------------------------------- 2. composite-null calibration
def tail_stat(cos_data, mean, cov, grid, P, seed):
    """One-sided excess of the data curve over its own fitted reference."""
    ref = K(occ.conditioned_pair_cos(mean, cov, P, seed=seed), grid)
    return float(np.max(K(cos_data, grid) - ref))


def calibration(n=2000, d=96, P=50_000, outer=40, inner=99, alpha=0.05, seed=10):
    """FPR of the refitted parametric-bootstrap test on cone-shaped nulls."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.55, 0.95, 25)          # the crowding-relevant tail
    rejections = 0
    for o in range(outer):
        X = cone(n, d, 0.577, rng)              # H0: pure cone, no crowding
        mean, cov = fit(X)
        t0 = tail_stat(pair_cos(X, P, rng), mean, cov, grid, P, seed=1000 + o)
        # parametric bootstrap WITH refitting
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 0, None)
        exceed = 0
        for b in range(inner):
            G = rng.standard_normal((n, d))
            Xb = mean[None, :] + (G * np.sqrt(w)[None, :]) @ V.T
            Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
            mb, cb = fit(Xb)                    # REFIT inside the replicate
            tb = tail_stat(pair_cos(Xb, P, rng), mb, cb, grid, P,
                           seed=2000 + o * inner + b)
            if tb >= t0:
                exceed += 1
        p = (1 + exceed) / (1 + inner)
        if p <= alpha:
            rejections += 1
    fpr = rejections / outer
    print(f"== 2. composite-null calibration (refit inside each replicate)")
    print(f"   {outer} cone-null corpora, {inner} bootstrap replicates each, "
          f"alpha = {alpha}")
    print(f"   false-positive rate: {fpr:.3f}  "
          f"(binomial 95% CI half-width ~{1.96*np.sqrt(alpha*(1-alpha)/outer):.3f})")
    print()
    return fpr


# ------------------------------------------------- 3. power / absorption
def power(n=2000, d=96, P=50_000, inner=99, alpha=0.05, reps=10, seed=20):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.55, 0.95, 25)
    print("== 3. power of the refitted test, and reference absorption, by pocket mass")
    print(f"{'pocket':>8} {'spread':>7} {'detect rate':>12} {'ref K(0.85) rise':>17}")
    for frac in (0.01, 0.05, 0.10, 0.20):
        for spread in (0.02, 0.06):
            det, rise = 0, []
            for r in range(reps):
                X = add_pocket(cone(n, d, 0.577, rng), frac, spread, rng)
                mean, cov = fit(X)
                t0 = tail_stat(pair_cos(X, P, rng), mean, cov, grid, P,
                               seed=3000 + r)
                w, V = np.linalg.eigh(cov)
                w = np.clip(w, 0, None)
                exceed = 0
                for b in range(inner):
                    G = rng.standard_normal((n, d))
                    Xb = mean[None, :] + (G * np.sqrt(w)[None, :]) @ V.T
                    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
                    mb, cb = fit(Xb)
                    tb = tail_stat(pair_cos(Xb, P, rng), mb, cb, grid, P,
                                   seed=4000 + r * inner + b)
                    if tb >= t0:
                        exceed += 1
                if (1 + exceed) / (1 + inner) <= alpha:
                    det += 1
                ref = K(occ.conditioned_pair_cos(mean, cov, P, seed=5), grid)
                rise.append(float(ref[np.searchsorted(grid, 0.85)]))
            print(f"{frac:8.0%} {spread:7.2f} {det/reps:12.1%} {np.mean(rise):17.2e}")
    print()


if __name__ == "__main__":
    tracking()
    calibration()
    power()
