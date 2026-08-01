#!/usr/bin/env python3
"""Build the fixed measurement subset for the E6 encoder loop.

Samples a stratified (proportional per subset) set of rows from the stored base
embedding parquet, splits it into train / held-out, and writes:

  <out>/manifest.json   {"seed", "n", "heldout_frac", "subsets": {name: count},
                         "uuid": [...], "subset": [...], "split": ["train"|"heldout", ...]}
  <out>/base.npy        (n, d) float32 L2-normalized base vectors, manifest order

The manifest order is THE canonical order for every round: re-embedded vectors,
weights, guards, and comparisons are all aligned to it. The held-out slice is
never used for mining, sampling weights, or training — verdicts only.

  python make_subset.py --emb-dir ~/ambit-legal/emb-1024-1M --out subset200k \
      --n 200000 --heldout-frac 0.2 --seed 0
"""
import argparse, glob, json, os

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.emb_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet shards in {a.emb_dir}")

    # pass 1: per-subset totals (metadata only, no vectors)
    counts, shard_meta = {}, []
    for f in files:
        t = pq.read_table(f, columns=["subset"])
        labs = t.column("subset").to_pylist()
        shard_meta.append((f, labs))
        for s in labs:
            counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    n = min(a.n, total)
    quota = {s: int(round(n * c / total)) for s, c in counts.items()}
    # rounding drift onto the largest subset
    drift = n - sum(quota.values())
    quota[max(counts, key=counts.get)] += drift
    print(f"{total} rows in {len(files)} shards; sampling {n}: {quota}", flush=True)

    # deterministic per-subset row selection over the global (shard-order) index
    rng = np.random.default_rng(a.seed)
    per_subset_rows = {}
    offsets, gidx = {}, 0
    subset_positions = {s: [] for s in counts}
    for f, labs in shard_meta:
        for s in labs:
            subset_positions[s].append(gidx)
            gidx += 1
    for s, pos in subset_positions.items():
        pos = np.asarray(pos)
        per_subset_rows[s] = set(pos[rng.choice(len(pos), quota[s], replace=False)])

    # pass 2: pull the selected vectors + ids in shard order (= manifest order)
    uuids, subsets, vecs = [], [], []
    gidx = 0
    for f, labs in shard_meta:
        t = pq.read_table(f, columns=["uuid", "subset", "embedding"])
        us = t.column("uuid").to_pylist()
        v = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        v = v.astype(np.float32).reshape(len(us), -1)
        take = [i for i, s in enumerate(labs) if gidx + i in per_subset_rows[s]]
        if take:
            uuids.extend(us[i] for i in take)
            subsets.extend(labs[i] for i in take)
            vecs.append(v[take])
        gidx += len(labs)
    X = np.vstack(vecs)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    split = np.array(["train"] * len(uuids), dtype=object)
    hold = rng.choice(len(uuids), int(a.heldout_frac * len(uuids)), replace=False)
    split[hold] = "heldout"

    np.save(os.path.join(a.out, "base.npy"), X)
    json.dump({"seed": a.seed, "n": len(uuids), "heldout_frac": a.heldout_frac,
               "subsets": quota, "uuid": uuids, "subset": subsets,
               "split": split.tolist()},
              open(os.path.join(a.out, "manifest.json"), "w"))
    print(f"wrote {a.out}/manifest.json + base.npy {X.shape} "
          f"({(split == 'heldout').sum()} held out)", flush=True)


if __name__ == "__main__":
    main()
