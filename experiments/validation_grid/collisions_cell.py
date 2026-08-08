#!/usr/bin/env python3
"""Per-document expected collision counts for a cell map, on GPU.

The label-free per-item readout for H2: for every document, the sum of
Gaussian flip probabilities against all other documents at noise scale
sigma (the cell's measured sigma*). Exact blocked computation — no
sub-quadratic approximation — feasible at millions of rows on GPU.

  python collisions_cell.py --map ~/maps/validation/trec-covid/bge-large \
    --sigma 0.12 --out ~/validation-runs/trec-covid.bge-large.cc.npz
"""
import argparse, glob, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--col-block", type=int, default=1 << 19,
                    help="column chunk for the row-sum accumulation; bounds "
                         "the float64 temporary at block x col-block x 8 B")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import torch

    us, vs = [], []
    for f in sorted(glob.glob(os.path.join(a.map, "*.parquet"))):
        t = pq.read_table(f, columns=["uuid", "embedding"])
        us.extend(t.column("uuid").to_pylist())
        v = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        vs.append(v.astype(np.float32).reshape(len(t), -1))
    X = np.vstack(vs)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Xt = torch.tensor(X, device=a.device)
    n = len(X)
    s2 = 2.0 * a.sigma
    cc = torch.zeros(n, dtype=torch.float64, device=a.device)
    inv = -1.0 / (s2 * np.sqrt(2.0))
    for s in range(0, n, a.block):
        e = min(s + a.block, n)
        acc = torch.zeros(e - s, dtype=torch.float64, device=a.device)
        # column-chunked so the float64 reduction temporary stays bounded:
        # summing a (block x n) float32 buffer with dtype=float64 would
        # materialize the whole block in float64 (39.9 GiB at n = 5.2M).
        for c0 in range(0, n, a.col_block):
            c1 = min(c0 + a.col_block, n)
            # single (block x col-block) buffer, chained in place:
            # gram -> chord -> phi
            g = Xt[s:e] @ Xt[c0:c1].T
            g.clamp_(-1.0, 1.0).mul_(-2.0).add_(2.0).clamp_min_(0).sqrt_()
            g.mul_(inv)
            torch.erf_(g)
            g.add_(1.0).mul_(0.5)
            lo, hi = max(s, c0), min(e, c1)          # self-pairs, if in chunk
            if lo < hi:
                g[torch.arange(lo - s, hi - s, device=a.device),
                  torch.arange(lo - c0, hi - c0, device=a.device)] = 0.0
            acc += g.sum(1, dtype=torch.float64)
            del g
        cc[s:e] = acc
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, uuid=np.array(us), cc=cc.cpu().numpy(),
             sigma=a.sigma)
    print(f"collisions saved: n={n}, sigma={a.sigma}, "
          f"median={float(np.median(cc.cpu().numpy())):.4f}", flush=True)


if __name__ == "__main__":
    main()
