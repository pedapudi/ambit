"""The anisotropy-conditioned reference must reproduce a mean-direction cone.

This is the calibration test that the original centered-spectrum ACG failed:
against a one-sided cone (the usual learned-embedding geometry), the ACG's
antipodal symmetry leaves its expected pair cosine at zero and its K(t)
essentially at the uniform null, so "data - reference" attributed the whole
cone to crowding. The mean-aware reference must track the cone; a planted
pocket must still stand out above it.
"""
import numpy as np

from ambit import occupancy as occ


def _cone(n, d, strength, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d)
    mu /= np.linalg.norm(mu)
    X = strength * np.sqrt(d) * mu + rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _pair_cos(X, P=100_000, seed=1):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), P)
    j = rng.integers(0, len(X), P)
    k = i != j
    return np.einsum("ij,ij->i", X[i[k]], X[j[k]])


def _fit(X):
    mean = X.mean(0)
    cov = (X.T @ X - len(X) * np.outer(mean, mean)) / (len(X) - 1)
    return mean, cov


def test_conditioned_reference_reproduces_a_mean_cone():
    X = _cone(4000, 128, 0.577, seed=0)          # cos(x, mu) ~ 0.5
    data = _pair_cos(X)
    mean, cov = _fit(X)
    ref = occ.conditioned_pair_cos(mean, cov, 100_000, seed=2)
    for t in (0.1, 0.2, 0.3):
        kd = float(np.mean(data >= t))
        kr = float(np.mean(ref >= t))
        assert abs(kd - kr) < 0.05, (t, kd, kr)


def test_centered_acg_cannot_see_the_cone():
    """Documents why the mean-aware reference exists: the centered-spectrum ACG
    misses nearly all of a mean cone's pair mass."""
    X = _cone(4000, 128, 0.577, seed=0)
    data = _pair_cos(X)
    _, cov = _fit(X)
    acg = occ.acg_pair_cos(np.linalg.eigvalsh(cov), 100_000, seed=2)
    kd = float(np.mean(data >= 0.2))
    ka = float(np.mean(acg >= 0.2))
    assert kd > 0.5 and ka < 0.1 * kd, (kd, ka)


def test_conditioned_reference_matches_acg_on_zero_mean_ellipse():
    """With no mean direction the two references must agree: the mean-aware fit
    generalizes the ACG rather than replacing its valid regime."""
    rng = np.random.default_rng(3)
    d = 128
    scales = np.ones(d)
    scales[:8] = 6.0                              # strong zero-mean ellipse
    X = rng.standard_normal((4000, d)) * scales
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    mean, cov = _fit(X)
    ref = occ.conditioned_pair_cos(mean, cov, 100_000, seed=4)
    acg = occ.acg_pair_cos(np.linalg.eigvalsh(cov), 100_000, seed=4)
    for t in (0.2, 0.4, 0.6):
        assert abs(float(np.mean(ref >= t)) - float(np.mean(acg >= t))) < 0.03


def test_planted_pocket_still_stands_out_above_the_fit():
    """The fitted reference must not absorb a planted pocket: excess above the
    reference at the pocket's similarity must survive fitting on the full
    (pocket-included) corpus."""
    X = _cone(4000, 128, 0.577, seed=5)
    rng = np.random.default_rng(6)
    c = rng.standard_normal(128)
    c /= np.linalg.norm(c)
    k = 200                                       # 5% pocket, near-duplicate tight
    P = c + 0.02 * rng.standard_normal((k, 128))
    X[-k:] = P / np.linalg.norm(P, axis=1, keepdims=True)
    data = _pair_cos(X, P=200_000)
    mean, cov = _fit(X)
    ref = occ.conditioned_pair_cos(mean, cov, 200_000, seed=7)
    t = 0.85                                      # pocket scale
    kd = float(np.mean(data >= t))
    kr = float(np.mean(ref >= t))
    expected_pocket_share = (k / 4000) ** 2       # ~2.5e-3 of pairs
    assert kd > 0.5 * expected_pocket_share
    assert kd > 10 * max(kr, 1e-9), (kd, kr)
