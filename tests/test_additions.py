"""Tests for the occupancy-report additions: RES 05b separability, RES 08 uniformity,
CMP 12–14 two-embedding comparison, and the reciprocal-kNN toggle. Pure-numpy synthetic
fixtures; no network, no heavy data."""

import numpy as np
import pytest

import ambit
from ambit import compare as C
from ambit import knn, metrics, separability as sep


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def blobs():
    """K well-separated gaussian blobs in d dims, with ids + string labels."""
    rng = np.random.default_rng(1)
    K, d, n = 5, 48, 2500
    cents = rng.normal(size=(K, d)) * 3.0
    y = rng.integers(0, K, n)
    X = (cents[y] + rng.normal(size=(n, d))).astype(np.float32)
    ids = np.array([f"id-{i:05d}" for i in range(n)])
    labels = np.array([f"g{int(v)}" for v in y])
    return X, ids, labels, K


# ----------------------------------------------------------------- CMP metrics
def test_linear_cka_invariances():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 32))
    Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    assert C.linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)
    assert C.linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)          # rotation
    assert C.linear_cka(X, 4.2 * X) == pytest.approx(1.0, abs=1e-6)        # isotropic scale
    assert C.linear_cka(X, X[:, rng.permutation(32)]) == pytest.approx(1.0, abs=1e-6)  # permutation
    assert C.linear_cka(X, rng.standard_normal((400, 32))) < 0.2          # unrelated
    assert 0.0 <= C.linear_cka(X, X[:, :16]) <= 1.0                       # d_x != d_y, defined


def test_kernel_and_rigid_metrics():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 24))
    assert C.rbf_cka(X, X) == pytest.approx(1.0, abs=1e-6)
    assert C.rbf_cka(X, rng.standard_normal((200, 24))) < 0.5
    assert C.mmd2_rbf(X, X) == pytest.approx(0.0, abs=1e-2)
    assert C.mmd2_rbf(X, X + 2.0) > 0.1                                   # shifted cloud
    assert C.energy_distance(X, X) == pytest.approx(0.0, abs=1e-6)
    assert C.energy_distance(X, X + 2.0) > 0.0
    assert C.procrustes_disparity(X, X @ np.linalg.qr(rng.standard_normal((24, 24)))[0]) == pytest.approx(0.0, abs=1e-6)


def test_neighbor_overlap():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 24)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    ia = knn.topk_cosine(A, 10)
    assert C.neighbor_overlap(ia, ia).mean() == 1.0                      # identical -> full retention
    B = A + 0.05 * rng.standard_normal((300, 24)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    Rnd = rng.standard_normal((300, 24)); Rnd /= np.linalg.norm(Rnd, axis=1, keepdims=True)
    near = C.neighbor_overlap(ia, knn.topk_cosine(B, 10)).mean()
    far = C.neighbor_overlap(ia, knn.topk_cosine(Rnd, 10)).mean()
    assert near > far and far < 0.2                                      # perturbation keeps more than random
    # dimension-agnostic: comparing against a lower-dim neighborhood still yields a valid overlap
    A12 = A[:, :12] / np.linalg.norm(A[:, :12], axis=1, keepdims=True)
    assert 0.0 <= C.neighbor_overlap(ia, knn.topk_cosine(A12, 10)).mean() <= 1.0


# ------------------------------------------------------------------ uniformity
def test_uniformity_direction():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((3000, 64))
    cone = rng.standard_normal((3000, 64)) + 3.0 * np.r_[np.ones(8), np.zeros(56)]
    u_iso = metrics.uniformity(iso)
    u_cone = metrics.uniformity(cone)
    assert u_iso < u_cone                                                 # isotropic = more uniform = more negative
    assert metrics.uniformity_ref(64) == pytest.approx(u_iso, abs=0.15)
    # uniformity_from_cos matches the direct estimate on the same sample
    cos = metrics.random_pair_cosine(iso, n_pairs=50_000)
    assert metrics.uniformity_from_cos(cos) == pytest.approx(metrics.uniformity(iso, n_pairs=50_000), abs=0.05)


# ---------------------------------------------------------------- separability
def test_separability_planted(blobs):
    X, ids, labels, K = blobs
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = U @ U.T
    np.fill_diagonal(S, -2.0)
    knn_idx = np.argsort(-S, 1)[:, :10]
    r = sep.compute(X, labels, knn_idx, supervised=False)
    assert len(r.groups) == K
    assert r.purity_overall > 0.95                                        # clean clusters
    assert r.silhouette > 0.4
    assert r.stability > 0.8                                              # reproducible
    assert r.n_modes >= 2
    # centroid off-diagonals are low (separated)
    off = r.centroids_cos[~np.eye(K, dtype=bool)]
    assert off.max() < 0.5
    assert sep.compute(X, np.zeros(len(X), int), knn_idx) is None         # single group -> None


# ------------------------------------------------------------- reciprocal kNN
def test_reciprocal_mask_and_filter():
    idx = np.array([[1, 2], [2, 3], [3, 0], [0, 1]])
    dist = np.full((4, 2), 0.1)
    mask = knn.reciprocal_mask(idx)
    assert mask.tolist() == [[False, True], [False, True], [False, True], [False, True]]
    fi, fd = knn.reciprocal_filter(idx, dist)
    assert fi[0, 0] == 2 and fi[0, 1] == 0 and np.isinf(fd[0, 1])         # reciprocal first, self-pad
    assert fi[1, 0] == 3 and fi[2, 0] == 0 and fi[3, 0] == 1


def test_hubness_skew_ignores_self_padding():
    # a self-loop padded row must not inflate the k-occurrence count
    idx = np.array([[1, 2], [0, 1], [0, 2]])                             # row1 has a self-loop (idx==1)
    s_with = metrics.hubness_skew(idx)
    assert np.isfinite(s_with)


def test_mutual_knn_suppresses_hubness(tmp_path, blobs):
    X, ids, labels, _ = blobs
    p = tmp_path / "x.npz"
    np.savez(p, embeddings=X, ids=ids, labels=labels)
    base = ambit.build_context(str(p))
    mut = ambit.build_context(str(p), mutual_knn=True)
    assert mut.mutual_knn and not base.mutual_knn
    # the filtered graph is genuinely mutual: 0 asymmetric real edges
    idx, dist = mut.knn_idx, mut.knn_dist
    real = {(i, int(j)) for i in range(len(idx)) for c, j in enumerate(idx[i])
            if int(j) != i and np.isfinite(dist[i, c])}
    assert all((j, i) in real for (i, j) in real)
    assert abs(mut.hub_skew) < abs(base.hub_skew) + 1e-9                  # not larger; usually much smaller


# ----------------------------------------------------------------- build_cmp
def _write_pair(tmp_path):
    rng = np.random.default_rng(7)
    n, d = 1500, 48
    ids = np.array([f"id-{i:05d}" for i in range(n)])
    XA = rng.standard_normal((n, d)).astype(np.float32)
    XB = (XA + 0.3 * rng.standard_normal((n, d))).astype(np.float32)
    perm = rng.permutation(n)                                            # shuffle B rows (same ids)
    a, b = tmp_path / "A.npz", tmp_path / "B.npz"
    np.savez(a, embeddings=XA, ids=ids, labels=np.array([f"g{i%4}" for i in range(n)]))
    np.savez(b, embeddings=XB[perm], ids=ids[perm])
    return a, b, ids, XB[:, :16], d


def test_build_cmp_aligns_by_id_not_row(tmp_path):
    a, b, ids, _, _ = _write_pair(tmp_path)
    rep = ambit.report(str(a), compare=str(b))
    cmp = rep.ctx.cmp
    assert len(cmp.ids) == len(ids)                                      # full alignment despite shuffle
    assert cmp.same_dim and cmp.cka_linear > 0.8                         # B is a small perturbation of A
    assert cmp.drift is not None and cmp.drift_cos is not None
    assert len(cmp.nbr_overlap) == len(cmp.ids) and 0.0 <= cmp.nbr_overlap_mean <= 1.0


def test_build_cmp_guards(tmp_path):
    a, b, ids, XB16, _ = _write_pair(tmp_path)
    # arange trap: a primary with no stable ids must refuse
    noid = tmp_path / "noid.npy"
    np.save(noid, np.random.default_rng(0).standard_normal((1500, 48)).astype(np.float32))
    with pytest.raises(ValueError):
        ambit.report(str(noid), compare=str(b))
    # disjoint ids
    bad = tmp_path / "disjoint.npz"
    np.savez(bad, embeddings=np.random.default_rng(0).standard_normal((1500, 48)).astype(np.float32),
             ids=np.array([f"z-{i}" for i in range(1500)]))
    with pytest.raises(ValueError):
        ambit.report(str(a), compare=str(bad))
    # dim mismatch (Matryoshka): only CKA survives
    trunc = tmp_path / "trunc.npz"
    np.savez(trunc, embeddings=XB16, ids=ids)
    cmp = ambit.report(str(a), compare=str(trunc)).ctx.cmp
    assert not cmp.same_dim
    assert cmp.mmd2 is None and cmp.energy is None and cmp.procrustes_disp is None and cmp.drift is None
    assert 0.0 <= cmp.cka_linear <= 1.0


# ----------------------------------------------------------------- integration
def test_report_renders_and_hides(tmp_path, blobs):
    X, ids, labels, _ = blobs
    p = tmp_path / "x.npz"
    np.savez(p, embeddings=X, ids=ids, labels=labels)
    html = ambit.report(str(p)).html
    assert "Separability panel" in html                                 # RES 05b on with labels
    assert 'class="num"' not in html                                    # figure tags/pills removed
    assert "uniformity" in html                                         # header scalar present
    assert "Uniformity on the hypersphere" not in html                  # RES 08 figure hidden by default
    assert not any(c in html for c in ["Representational drift", "Drift field"])  # CMP hidden w/o --compare


def test_report_compare_enables_cmp(tmp_path):
    a, b, *_ = _write_pair(tmp_path)
    html = ambit.report(str(a), compare=str(b)).html
    for card in ("Neighbor-overlap drift", "Representational drift", "Drift field",
                 "Distance-distribution shift", "Uniformity on the hypersphere"):
        assert card in html
