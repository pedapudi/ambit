#!/usr/bin/env python3
"""Loop plumbing dry-run + the "adapter" rung of the capacity ladder.

Trains a linear adapter (identity-initialized d x d map) directly on the stored
base vectors with exactly the machinery train_ambit.py uses on the encoder —
resolution-weighted batches, guard_mask, confusion + preservation losses — and
writes the adapted vectors aligned to the manifest, ready for measure_round.py.

CPU-only; validates the whole round pipeline end-to-end before any GPU work,
and on the full subset it *is* the real-data adapter baseline for the TR's
capacity ladder (adapter < LoRA < full fine-tune).

  python dry_run_adapter.py --subset subset200k --sigma 0.1226 \
      --steps 600 --out rounds/adapter.npy
"""
import argparse, json, os, time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2048,
                    help="pair losses need large in-batch pair statistics: at "
                         "batch 256 the map overfits in-batch pairs and destroys "
                         "global geometry (held-out overlap 0.82); at 2048 it is "
                         "well-behaved")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-p", type=float, default=1.0)
    ap.add_argument("--guard-top-m", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    from ambit import knn
    from ambit import training as tr

    torch.manual_seed(a.seed)
    manifest = json.load(open(os.path.join(a.subset, "manifest.json")))
    base = np.load(os.path.join(a.subset, "base.npy")).astype(np.float32)
    train_rows = np.flatnonzero(np.asarray(manifest["split"], object) == "train")
    d = base.shape[1]

    w = tr.resolution_weights(base[train_rows], a.sigma, floor=0.25)
    topk = knn.topk_cosine(base[train_rows], a.guard_top_m)
    Xt = torch.tensor(base[train_rows])

    W = torch.eye(d, requires_grad=True)
    opt = torch.optim.Adam([W], lr=a.lr)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for step in range(a.steps):
        rows = rng.choice(len(train_rows), a.batch, replace=False, p=w)
        xb = Xt[rows]
        z = xb @ W
        guard = torch.tensor(tr.guard_mask(topk, rows))
        loss = (tr.confusion_loss(z, a.sigma, exclude=guard)
                + a.lambda_p * tr.preservation_loss(z, xb))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{a.steps} loss {float(loss):.5f} "
                  f"({(step+1)*a.batch/(time.time()-t0):.0f} ex/s)", flush=True)

    with torch.no_grad():
        Z = (torch.tensor(base) @ W).numpy()
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out, Z.astype(np.float32))
    print(f"wrote {a.out} {Z.shape}", flush=True)


if __name__ == "__main__":
    main()
