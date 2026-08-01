"""Tests for the training-time regularizers and mining utilities."""

import numpy as np
import pytest

from ambit import occupancy as occ
from ambit import training as tr

torch = pytest.importorskip("torch", reason="torch optional; regularizers are torch-first")


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _clumped(n, d, k=60, spread=0.03, seed=0):
    rng = np.random.default_rng(seed)
    X = _unit(rng.standard_normal((n - k, d)))
    c = _unit(rng.standard_normal((1, d)))
    return np.vstack([X, _unit(c + spread * rng.standard_normal((k, d)))])


# ---------------------------------------------------------------- losses
def test_confusion_loss_numpy_matches_expected_collisions():
    X = _unit(np.random.default_rng(0).standard_normal((64, 16)))
    sigma = 0.3
    loss = tr.confusion_loss(X, sigma)
    # brute force over ordered pairs
    g = np.clip(X @ X.T, -1, 1)
    r = np.sqrt(np.clip(2 - 2 * g, 0, None))
    p = occ._phi(-r / (2 * sigma))
    keep = ~np.eye(len(X), dtype=bool)
    assert abs(loss - p[keep].mean()) < 1e-9


def test_confusion_loss_torch_matches_numpy_and_is_differentiable():
    X = _clumped(48, 16, k=12, seed=1)
    zn = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    lt = tr.confusion_loss(zn, 0.25)
    ln = tr.confusion_loss(X, 0.25)
    assert abs(float(lt.detach()) - ln) < 1e-6
    lt.backward()
    assert zn.grad is not None and torch.isfinite(zn.grad).all()


def test_confusion_gradient_lives_in_the_window():
    # far pairs get (exponentially) no gradient; close pairs get pushed
    d = 16
    a = np.zeros(d); a[0] = 1.0
    near = _unit(np.array([a + 0.1 * np.eye(d)[1]]))[0]
    far = np.zeros(d); far[1] = 1.0                                # 90 degrees away
    z = torch.tensor(np.stack([a, near, far]), dtype=torch.float64, requires_grad=True)
    loss = tr.confusion_loss(z, 0.1)
    loss.backward()
    g_near = float(z.grad[1].norm())
    g_far = float(z.grad[2].norm())
    assert g_near > 1e3 * max(g_far, 1e-30)


def test_confusion_exclude_mask():
    X = _clumped(32, 16, k=8, seed=2)
    full = tr.confusion_loss(X, 0.3)
    # excluding the clump's own pairs must lower the penalty
    m = np.zeros((32, 32), bool)
    m[24:, 24:] = True
    part = tr.confusion_loss(X, 0.3, exclude=m)
    assert part < full


def test_preservation_loss_zero_at_reference_and_positive_off_it():
    X = _unit(np.random.default_rng(3).standard_normal((40, 16)))
    assert tr.preservation_loss(X, X) < 1e-10
    Y = _unit(X + 0.3 * np.random.default_rng(4).standard_normal(X.shape))
    assert tr.preservation_loss(Y, X) > 1e-3
    zt = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    ref = torch.tensor(X, dtype=torch.float64)
    l = tr.preservation_loss(zt, ref)
    assert abs(float(l) - tr.preservation_loss(Y, X)) < 1e-6
    l.backward()
    assert torch.isfinite(zt.grad).all()


def test_uniformity_loss_prefers_spread():
    rng = np.random.default_rng(5)
    spread = _unit(rng.standard_normal((64, 16)))
    tight = _unit(spread[0] + 0.05 * rng.standard_normal((64, 16)))
    assert tr.uniformity_loss(spread) < tr.uniformity_loss(tight)


def test_training_step_reduces_collisions_and_preserves_structure():
    # one-variable sanity: optimizing conf+preserve on a clumped batch lowers
    # expected collisions without wrecking the reference structure
    X = _clumped(96, 24, k=24, spread=0.02, seed=6)
    ref = torch.tensor(X, dtype=torch.float64)
    z = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.03)
    c0 = tr.confusion_loss(X, 0.2)
    for _ in range(120):
        opt.zero_grad()
        loss = tr.confusion_loss(z, 0.2) + 0.5 * tr.preservation_loss(z, ref)
        loss.backward()
        opt.step()
    zf = z.detach().numpy()
    assert tr.confusion_loss(zf, 0.2) < 0.5 * c0                    # collisions down
    assert tr.preservation_loss(zf, X) < 0.1                        # structure held


# ---------------------------------------------------------------- mining
def test_guard_mask_excludes_base_neighbors_from_the_penalty():
    from ambit import knn

    X = _clumped(64, 16, k=16, spread=0.02, seed=9).astype(np.float32)
    Xn = _unit(X)
    topk = knn.topk_cosine(Xn, 5)
    rows = np.arange(40, 64)                     # batch spanning the clump
    m = tr.guard_mask(topk, rows)
    assert m.shape == (24, 24) and m.dtype == bool
    assert (m == m.T).all() and not m.diagonal().any()
    # every guarded pair really is a base top-5 relation in one direction
    ii, jj = np.nonzero(m)
    for i, j in zip(ii[:50], jj[:50]):
        gi, gj = rows[i], rows[j]
        assert gj in topk[gi] or gi in topk[gj]
    # guarding the clump's own neighbor pairs must lower the penalty, and the
    # guarded pairs must contribute no gradient
    full = tr.confusion_loss(X[rows], 0.2)
    part = tr.confusion_loss(X[rows], 0.2, exclude=m)
    assert part < full
    z = torch.tensor(X[rows], dtype=torch.float64, requires_grad=True)
    everything = torch.ones(len(rows), len(rows), dtype=torch.bool)
    l = tr.confusion_loss(z, 0.2, exclude=everything)
    l.backward()
    assert float(l) == 0.0 and float(z.grad.abs().max()) == 0.0


def test_resolution_weights_oversample_the_clump():
    X = _clumped(400, 24, k=80, seed=7)
    w = tr.resolution_weights(X, 0.2, floor=0.2)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w[-80:].mean() > 3 * w[:-80].mean()


def test_miner_respects_window_and_guard():
    X = _clumped(300, 24, k=60, spread=0.05, seed=8).astype(np.float32)
    a, nb = tr.mine_confusable_negatives(X, cos_window=(0.5, 0.95),
                                         guard_top_m=3, per_anchor=4, seed=0)
    assert len(a) == len(nb) and len(a) > 0
    Xn = _unit(X)
    cos = np.einsum("ij,ij->i", Xn[a], Xn[nb])
    assert (cos >= 0.5 - 1e-6).all() and (cos <= 0.95 + 1e-6).all()
    # guard: no mined negative is its anchor's top-3 neighbor
    for aa, nn in zip(a[:200], nb[:200]):
        sims = Xn @ Xn[aa]
        sims[aa] = -2
        top3 = np.argpartition(-sims, 3)[:3]
        assert nn not in top3

    with pytest.raises(ValueError):
        tr.mine_confusable_negatives(X, cos_window=(0.9, 0.5))
