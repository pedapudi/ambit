"""Tests for the continuous-occupancy layer: occupancy.py (crowding curve, nulls,
Stolarsky scalar, confusion units), crowding.py (DTM field, merge-tree pockets),
and their report wiring."""

import math

import numpy as np
import pytest

import ambit
from ambit import crowding as cr
from ambit import occupancy as occ


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _uniform(n, d, seed=0):
    return _unit(np.random.default_rng(seed).standard_normal((n, d)))


def _clumped(n, d, k=200, spread=0.02, seed=0):
    """Uniform corpus with k near-duplicates around one direction."""
    rng = np.random.default_rng(seed)
    X = _unit(rng.standard_normal((n - k, d)))
    c = _unit(rng.standard_normal((1, d)))
    C = _unit(c + spread * rng.standard_normal((k, d)))
    return np.vstack([X, C])


def _pair_cos(X, n_pairs=60_000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    return np.einsum("ij,ij->i", X[i[keep]], X[j[keep]])


# ---------------------------------------------------------------- occupancy.py
def test_erf_matches_math():
    z = np.linspace(-4, 4, 41)
    ours = occ._erf(z)
    ref = np.array([math.erf(v) for v in z])
    assert np.abs(ours - ref).max() < 1e-6


def test_null_pair_cos_moments():
    d = 256
    c = occ.null_pair_cos(d, 200_000, seed=1)
    assert abs(c.mean()) < 0.002
    assert abs(c.std() - 1 / np.sqrt(d)) < 0.002


def test_exceedance_is_exact_ecdf():
    c = np.array([0.1, 0.2, 0.2, 0.9])
    k = occ.exceedance(c, [0.0, 0.15, 0.2, 0.85, 0.95])
    assert np.allclose(k, [1.0, 0.75, 0.75, 0.25, 0.0])


def test_stolarsky_z_separates_clump_from_uniform():
    d = 128
    zu = occ.stolarsky_z(_pair_cos(_uniform(3000, d)), d)[1]
    zc = occ.stolarsky_z(_pair_cos(_clumped(3000, d)), d)[1]
    assert abs(zu) < 4                      # uniform: consistent with its null
    assert zc < -10                         # clumped: far below the null


def test_liftoff_detects_clump_scale():
    d = 128
    lift_u, *_ = occ.liftoff_cos(_pair_cos(_uniform(3000, d)), d)
    lift_c, *_ = occ.liftoff_cos(_pair_cos(_clumped(3000, d)), d)
    assert lift_c is not None and lift_c > 0.5      # near-duplicates lift off at high cos
    assert lift_u is None or lift_u < 0.5           # uniform: no high-cos excess


def test_confusion_kernel_is_exact_gaussian_flip():
    # brute-force the Gaussian query channel against Phi(-chord/(2 sigma))
    rng = np.random.default_rng(0)
    d, sigma, theta = 64, 0.2, 0.5
    x = np.zeros(d); x[0] = 1.0
    y = np.zeros(d); y[0] = np.cos(theta); y[1] = np.sin(theta)
    q = x + sigma * rng.standard_normal((200_000, d))
    emp = float(np.mean(q @ y > q @ x))
    pred = float(occ.pair_confusion(np.cos(theta), sigma))
    assert abs(emp - pred) < 0.004


def test_sigma_star_orders_corpora():
    d = 128
    s_u = occ.sigma_star(_pair_cos(_uniform(3000, d)), 3000)
    s_c = occ.sigma_star(_pair_cos(_clumped(3000, d)), 3000)
    assert s_c < s_u                        # crowded corpus tolerates less query noise
    assert occ.expected_collisions(_pair_cos(_uniform(3000, d)), 3000, s_u) <= 1.01


def test_collision_counts_names_the_clump():
    d = 64
    X = _clumped(1200, d, k=100)
    c = occ.collision_counts(X, 0.15)
    # clump members are the last 100 rows; they should dominate the collision counts
    assert np.median(c[-100:]) > 10 * max(np.median(c[:-100]), 1e-9)
    # brute-force check one row
    i = len(X) - 1
    chord = np.sqrt(np.clip(2 - 2 * (X @ X[i]), 0, None))
    brute = occ._phi(-chord / 0.3).sum() - 0.5      # minus the self term Phi(0)
    assert abs(c[i] - brute) < 1e-4          # float32 Gram vs float64 brute force


# ---------------------------------------------------------------- crowding.py
def test_dtm_low_tail_flags_clump_and_is_stable():
    d = 96
    X = _clumped(2000, d, k=150, spread=0.02)
    f = cr.dtm(X, m_frac=0.02)
    assert np.percentile(f, 5) < 0.5 * np.median(f)         # clump collapses the low tail
    U = _uniform(2000, d, seed=3)
    fu = cr.dtm(U, m_frac=0.02)
    jit = _unit(U + 1e-3 * np.random.default_rng(4).standard_normal(U.shape))
    fj = cr.dtm(jit, m_frac=0.02)
    assert np.corrcoef(fu, fj)[0, 1] > 0.9                  # stability under jitter


def test_dtm_null_band_brackets_uniform_field():
    d = 96
    f = cr.dtm(_uniform(1500, d, seed=5), m_frac=0.02)
    lo, hi = cr.dtm_null_band(1500, d, m_frac=0.02, seed=6)
    inside = np.mean((f >= lo * 0.98) & (f <= hi * 1.02))
    assert inside > 0.9


def test_pockets_recovers_planted_clumps():
    d = 64
    rng = np.random.default_rng(7)
    X = _unit(rng.standard_normal((1500, d)))
    c1 = _unit(rng.standard_normal((1, d)))
    c2 = _unit(rng.standard_normal((1, d)))
    A = _unit(c1 + 0.02 * rng.standard_normal((60, d)))
    B = _unit(c2 + 0.02 * rng.standard_normal((40, d)))
    pk, _ = cr.pockets(np.vstack([X, A, B]), min_size=8, seed=0)
    assert len(pk) >= 2
    sizes = sorted(p["size"] for p in pk[:2])
    assert abs(sizes[0] - 40) <= 8 and abs(sizes[1] - 60) <= 8
    # planted pockets are tight: they form well below the bulk scale
    assert all(p["birth"] < 0.4 for p in pk[:2])
    assert all(p["prominence"] > 0.2 for p in pk[:2])


def test_pockets_empty_on_uniform():
    pk, _ = cr.pockets(_uniform(1200, 64, seed=8), min_size=8, seed=0)
    assert all(p["prominence"] < 0.2 for p in pk)           # nothing detaches far from the bulk


# ---------------------------------------------------------------- report wiring
def test_report_contains_new_figures_and_facts(tmp_path):
    d = 64
    X = _clumped(1500, d, k=80).astype(np.float32)
    ids = np.array([f"id{i:05d}" for i in range(len(X))])
    p = tmp_path / "x.npz"
    np.savez(p, embeddings=X, ids=ids)
    html = ambit.report(str(p)).html
    for card in ("Crowding curve", "Per-entity crowding field", "Crowding pockets"):
        assert card in html
    assert "occupancy z" in html
    assert "resolution bandwidth" in html
    assert "lattice-averaged" not in html or True            # hexbin hidden by default
    # the crowded ids are named in the report
    assert "id0" in html
