"""Tier-3 validation: E3 scaling curves.

Time and peak memory versus corpus size and dimension, separated by pipeline
phase (streaming scan / context build / continuous-layer statistics), plus the
reservoir-vs-full tradeoff and the sorted-shards failure mode for approximate
scans. Each measurement runs in a forked child so peak-RSS is per-phase.

Generates corpora as .npy via chunked memmap writes (disk-frugal: each corpus
is deleted after its measurements). Run on a large-disk host:
  python experiments/tier3_scaling.py --out results --work /tmp/e3work
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import shutil
import time

import numpy as np

N_SWEEP = [10_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
D_SWEEP = [128, 256, 512, 1024, 2048, 4096]
D_FIX, N_FIX = 1024, 1_000_000


def write_corpus(path, n, d, pocket_frac=0.05, seed=0, chunk=100_000):
    """Chunked unit-normalized Gaussian corpus with a planted pocket in the
    LAST rows (so a sorted/sharded approximate scan can miss it)."""
    rng = np.random.default_rng(seed)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(n, d))
    k = int(pocket_frac * n)
    c = rng.standard_normal((1, d))
    c /= np.linalg.norm(c)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        X = rng.standard_normal((e - s, d))
        lo = max(s, n - k)
        if lo < e:                                  # pocket rows in this chunk
            m = e - lo
            X[lo - s:] = c + 0.01 * rng.standard_normal((m, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        arr[s:e] = X.astype(np.float32)
    arr.flush()
    del arr


def _phase_child(q, phase, path, kwargs):
    import ambit
    from ambit import crowding as cr
    from ambit import metrics, occupancy as occ
    t0 = time.time()
    if phase == "scan":
        sc = ambit.scan(path, **kwargs)
        extra = {"n": int(sc.n)}
    elif phase == "ctx":
        sc = ambit.scan(path, **kwargs)
        t0 = time.time()                            # time ctx only
        ambit.build_ctx(sc)
        extra = {}
    elif phase == "continuous":
        sc = ambit.scan(path, **kwargs)
        es = sc.sample.normalize()
        pc = metrics.random_pair_cosine(es.X, n_pairs=200_000,
                                        normalized=True, seed=0)
        t0 = time.time()
        occ.rank_envelope(pc, int(sc.dim), reps=99, seed=0)
        occ.stolarsky_z(pc, int(sc.dim), seed=0)
        occ.sigma_star(pc, int(sc.n))
        sub = es.X if len(es.X) <= 6000 else es.X[
            np.random.default_rng(0).choice(len(es.X), 6000, replace=False)]
        cr.dtm(sub, m_frac=0.02)
        cr.pockets(es.X, min_size=8, seed=0)
        extra = {}
    elif phase == "approx":
        sc = ambit.scan(path, approx=kwargs.pop("approx_rows"), **kwargs)
        es = sc.sample.normalize()
        pc = metrics.random_pair_cosine(es.X, n_pairs=200_000,
                                        normalized=True, seed=0)
        p, lift, *_ = occ.rank_envelope(pc, int(sc.dim), reps=99, seed=0)
        extra = {"mean_cos": float(pc.mean()), "lift": lift, "p": p,
                 "scanned": int(sc.scanned)}
    wall = time.time() - t0
    q.put({"wall_s": wall,
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           **extra})


def measure(phase, path, **kwargs):
    ctx = mp.get_context("fork")                   # per-phase peak-RSS isolation
    q = ctx.Queue()
    p = ctx.Process(target=_phase_child, args=(q, phase, path, kwargs))
    p.start()
    out = q.get()
    p.join()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--work", default="e3work")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    results = {"n_sweep": [], "d_sweep": [], "approx": []}

    def free_gb():
        return shutil.disk_usage(a.work).free / 1e9

    # ---- n sweep at d = D_FIX -------------------------------------------
    for n in N_SWEEP:
        need = n * D_FIX * 4 / 1e9
        if need + 30 > free_gb():
            results["n_sweep"].append({"n": n, "skipped": f"disk ({need:.0f} GB needed)"})
            continue
        path = os.path.join(a.work, f"n{n}.npy")
        t0 = time.time()
        write_corpus(path, n, D_FIX, seed=1)
        gen_s = time.time() - t0
        row = {"n": n, "d": D_FIX, "gen_s": gen_s}
        for phase in ("scan", "ctx", "continuous"):
            row[phase] = measure(phase, path)
        results["n_sweep"].append(row)
        print(f"n={n}: {row}", flush=True)
        if n != N_FIX:
            os.remove(path)
        json.dump(results, open(os.path.join(a.out, "e3.json"), "w"), indent=1)

    # ---- approx / sorted-shards tradeoff on the 1e6 corpus --------------
    full_path = os.path.join(a.work, f"n{N_FIX}.npy")
    full = measure("approx", full_path, approx_rows=None) \
        if os.path.exists(full_path) else None
    if full:
        results["approx"].append({"rows": "full", **full})
        for rows in (50_000, 100_000, 300_000):
            r = measure("approx", full_path, approx_rows=rows)
            results["approx"].append({"rows": rows, **r})
            print(f"approx {rows}: {r}", flush=True)
        os.remove(full_path)
    json.dump(results, open(os.path.join(a.out, "e3.json"), "w"), indent=1)

    # ---- d sweep at n = N_FIX -------------------------------------------
    for d in D_SWEEP:
        need = N_FIX * d * 4 / 1e9
        if need + 30 > free_gb():
            results["d_sweep"].append({"d": d, "skipped": "disk"})
            continue
        path = os.path.join(a.work, f"d{d}.npy")
        t0 = time.time()
        write_corpus(path, N_FIX, d, seed=2)
        row = {"n": N_FIX, "d": d, "gen_s": time.time() - t0}
        for phase in ("scan", "ctx", "continuous"):
            row[phase] = measure(phase, path)
        results["d_sweep"].append(row)
        print(f"d={d}: {row}", flush=True)
        os.remove(path)
        json.dump(results, open(os.path.join(a.out, "e3.json"), "w"), indent=1)

    print("done", flush=True)


if __name__ == "__main__":
    main()
