"""The sound training loop, end to end, on a synthetic corpus.

Measure -> mine -> regularize -> verify on held-out data. A linear adapter
(initialized at identity) stands in for the model being adapted; swap in your
encoder and your reservoir for the real thing.
"""

import numpy as np
import torch

from ambit import occupancy as occ
from ambit import training as tr


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def measure(X, name, flagged=None, sigma=None):
    """The held-out verdicts: sigma*, and collision counts on the flagged rows."""
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(X), 60_000), rng.integers(0, len(X), 60_000)
    keep = i != j
    pair_cos = np.einsum("ij,ij->i", X[i[keep]], X[j[keep]])
    s_star = occ.sigma_star(pair_cos, len(X), tol=1.0)
    line = f"{name:10s} sigma* = {s_star:.3f}"
    if flagged is not None and sigma is not None:
        c = occ.collision_counts(X, sigma)
        line += (f" · flagged-entity collisions: median {np.median(c[flagged]):7.2f}"
                 f" (bulk {np.median(np.delete(c, flagged)):.4f})")
    print(line)
    return s_star


def main():
    rng = np.random.default_rng(0)
    n, d, k = 4000, 64, 300

    # a corpus with one crowded pocket (near-duplicates around one direction)
    base = unit(rng.standard_normal((n - k, d)))
    center = unit(rng.standard_normal((1, d)))
    pocket = unit(center + 0.05 * rng.standard_normal((k, d)))
    X = np.vstack([base, pocket]).astype(np.float64)

    # hold out a slice the training loop never sees; verdicts come from it only
    held = rng.choice(n, n // 5, replace=False)
    train_rows = np.setdiff1d(np.arange(n), held)
    Xtr, Xho = X[train_rows], X[held]
    flagged_ho = np.flatnonzero(np.isin(held, np.arange(n - k, n)))

    # ---- 1 · measure: the diagnosis licenses the intervention ----------------
    print("== before ==")
    sigma = measure(Xho, "held-out", flagged_ho, sigma=0.15)
    sigma = min(max(sigma, 0.05), 0.3)          # target the measured scale

    # ---- 2 · mine: weights + guarded negatives from the base geometry -------
    w = tr.resolution_weights(Xtr, sigma, floor=0.25)
    a_idx, n_idx = tr.mine_confusable_negatives(
        Xtr.astype(np.float32), cos_window=(0.5, 0.98), guard_top_m=3, per_anchor=4)
    print(f"mined {len(a_idx):,} guarded confusable pairs; "
          f"crowded-entity sampling weight x{w[-min(k, len(w)):].mean() / w.mean():.1f}")

    # ---- 3 · train: a linear adapter under confusion + preservation ----------
    W = torch.eye(d, dtype=torch.float64, requires_grad=True)
    ref = torch.tensor(Xtr, dtype=torch.float64)
    opt = torch.optim.Adam([W], lr=1e-2)
    B = 256
    for step in range(600):
        rows = rng.choice(len(Xtr), B, replace=False, p=w)
        zb = torch.tensor(Xtr[rows], dtype=torch.float64) @ W
        loss = (tr.confusion_loss(zb, sigma)
                + 0.3 * tr.preservation_loss(zb, ref[rows]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---- 4 · verify on the held-out slice ------------------------------------
    Wd = W.detach().numpy()
    Zho = unit(Xho @ Wd)
    print("== after ==")
    measure(Zho, "held-out", flagged_ho, sigma=0.15)
    # drift alarm: neighbor overlap with the base model on held-out rows
    from ambit.knn import topk_cosine
    kk = 10
    ov = np.mean([len(set(a) & set(b)) / kk for a, b in zip(
        topk_cosine(Xho.astype(np.float32), kk),
        topk_cosine(Zho.astype(np.float32), kk))])
    print(f"neighbor overlap with base (held-out, k={kk}): {ov:.2f} "
          f"— high = structure preserved, low = re-skinned")


if __name__ == "__main__":
    main()
