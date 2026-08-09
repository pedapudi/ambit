#!/usr/bin/env python3
"""Figure coordinates for one cell map: crowding curve, C(sigma), DTM CDF.

Emits pgfplots coordinate lists for the three multi-corpus overlays in the
technical report (the crowding curve, expected collisions vs. query noise,
and the distance-to-measure CDF). Written so a corpus added to the grid can
be added to those figures without hand-assembling numbers.

  python curve_data.py --map ~/ambit/evals/external-grid/maps/hotpotqa/BAAI-bge-large-en-v1.5 \
    --label HotpotQA --out hotpotqa-curves.txt

Sample sizes match the report's stated conventions: 2e5 pairs for the
curve and the collision integral, a capped reservoir for the DTM field.
"""
import argparse, glob, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ambit import crowding, occupancy  # noqa: E402


def load_map(map_dir, cap=None, seed=0):
    import pyarrow.parquet as pq
    vs = []
    for f in sorted(glob.glob(os.path.join(map_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["embedding"])
        v = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        vs.append(v.astype(np.float32).reshape(len(t), -1))
    X = np.vstack(vs)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if cap and len(X) > cap:
        X = X[np.random.default_rng(seed).choice(len(X), cap, replace=False)]
    return X


def pair_cos(X, n_pairs, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    return np.einsum("ij,ij->i", X[i[keep]], X[j[keep]])


def fmt(xs, ys, places=(4, 4)):
    return " ".join(f"({x:.{places[0]}f},{y:.{places[1]}f})" for x, y in zip(xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--label", required=True, help="corpus name for the block headers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--dtm-cap", type=int, default=16_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X = load_map(a.map)
    n, d = X.shape
    pc = pair_cos(X, a.pairs, a.seed)
    print(f"{a.label}: n={n} d={d} pairs={len(pc)}", flush=True)

    # --- crowding curve: log10 fraction of pairs at or above each threshold
    grid = np.linspace(0.0, 1.0, 80)
    exc = occupancy.exceedance(pc, grid)
    floor = 10 ** -5.602                       # the report's plotted floor
    kc = np.log10(np.maximum(exc, floor))

    # --- expected collisions vs. query noise (log-log)
    sig = np.logspace(-2.6, 0.1, 60)
    cc = np.array([occupancy.expected_collisions(pc, n, s) for s in sig])
    cx = np.log10(sig)
    cy = np.log10(np.maximum(cc, 1e-7))
    sstar = occupancy.sigma_star(pc, n)

    # --- DTM field CDF on a capped reservoir
    R = load_map(a.map, cap=a.dtm_cap, seed=a.seed)
    f = np.sort(crowding.dtm(R, m_frac=0.02))
    idx = np.linspace(0, len(f) - 1, 60).astype(int)
    dx, dy = f[idx], (idx + 1) / len(f)

    with open(a.out, "w") as fh:
        fh.write(f"%% K{a.label}\n{fmt(grid, kc, (4, 3))}\n\n")
        fh.write(f"%% C{a.label}\n{fmt(cx, cy, (4, 4))}\n\n")
        fh.write(f"%% Dtm{a.label}\n{fmt(dx, dy, (4, 3))}\n")
    print(f"sigma* = {sstar:.4f}  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
