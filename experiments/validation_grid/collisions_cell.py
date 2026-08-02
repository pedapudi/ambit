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
    ap.add_argument("--block", type=int, default=4096)
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
    for s in range(0, n, a.block):
        e = min(s + a.block, n)
        g = (Xt[s:e] @ Xt.T).clamp(-1, 1)
        r = (2.0 - 2.0 * g).clamp_min(0).sqrt()
        p = 0.5 * (1.0 + torch.erf((-r / s2) / np.sqrt(2.0)))
        p[torch.arange(e - s, device=a.device),
          torch.arange(s, e, device=a.device)] = 0.0
        cc[s:e] = p.sum(1, dtype=torch.float64)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, uuid=np.array(us), cc=cc.cpu().numpy(),
             sigma=a.sigma)
    print(f"collisions saved: n={n}, sigma={a.sigma}, "
          f"median={float(np.median(cc.cpu().numpy())):.4f}", flush=True)


if __name__ == "__main__":
    main()
