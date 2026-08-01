"""Tier-1 validation experiments (validation-plan Part I: E1a-e, E4c, E5).

Every experiment writes one JSON file to --out (default ./results). Pure numpy;
designed for a many-core CPU host. Seeds are explicit everywhere; nothing here
reads labels. Run:  python experiments/tier1_sensitivity.py --out results
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from ambit import crowding as cr
from ambit import occupancy as occ

D_BENCH, N_BENCH, K_BENCH, SPREAD = 768, 4000, 200, 0.01


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def make_corpus(n, d, k=0, spread=SPREAD, seed=0):
    rng = np.random.default_rng(seed)
    X = unit(rng.standard_normal((n, d))).astype(np.float32)
    if k:
        c = unit(rng.standard_normal((1, d)))
        X[-k:] = unit(c + spread * rng.standard_normal((k, d))).astype(np.float32)
    return X


def pair_cos(X, P=200_000, seed=0):
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(X), P), rng.integers(0, len(X), P)
    keep = i != j
    return np.einsum("ij,ij->i", X[i[keep]].astype(np.float64),
                     X[j[keep]].astype(np.float64))


def headline(X, d, pc=None, seed=0, dtm_cap=6000, corpus_n=None):
    """The report's headline readouts for one corpus/reservoir. `corpus_n` is
    the size of the corpus the reservoir represents — sigma* semantics use the
    corpus item count with the reservoir's pair distribution, exactly as the
    report's facts row does (it passes scan.n, not the reservoir size)."""
    pc = pair_cos(X, seed=seed) if pc is None else pc
    lift, *_ = occ.liftoff_cos(pc, d, seed=seed)
    _, z = occ.stolarsky_z(pc, d, seed=seed)
    ss = occ.sigma_star(pc, corpus_n or len(X))
    sub = X if len(X) <= dtm_cap else X[np.random.default_rng(seed).choice(
        len(X), dtm_cap, replace=False)]
    f = cr.dtm(sub, m_frac=0.02)
    pk, _ = cr.pockets(X, min_size=8, seed=0)
    top = pk[0] if pk else None
    return {"liftoff": lift, "z": z, "sigma_star": ss,
            "dtm_p1": float(np.percentile(f, 1)),
            "pocket": None if top is None else
            {"size": top["size"], "birth": top["birth"], "death": top["death"]}}


# ---------------------------------------------------------------- E1a
def e1a(out):
    """Reservoir-size sweep against a fixed 10^6-row corpus (5% pocket)."""
    n, d = 1_000_000, D_BENCH
    corpus = make_corpus(n, d, k=n // 20, seed=0)
    rows = []
    for n_r in (1000, 2000, 4000, 8000, 16000):
        for seed in range(10):
            rng = np.random.default_rng(1000 + seed)
            R = corpus[rng.choice(n, n_r, replace=False)]
            h = headline(R, d, seed=seed, dtm_cap=16000, corpus_n=n)
            rows.append({"n_r": n_r, "seed": seed, **h})
    ref = headline(corpus[np.random.default_rng(7).choice(n, 32000, replace=False)],
                   d, seed=7, dtm_cap=16000, corpus_n=n)
    json.dump({"rows": rows, "reference_32k": ref}, open(out, "w"), indent=1)


# ---------------------------------------------------------------- E1b
def e1b(out):
    """Pair-sample-size sweep: sd of headline scalars vs P, against 1/sqrt(P)."""
    X = make_corpus(N_BENCH, D_BENCH, K_BENCH, seed=0)
    rows = []
    for P in (10_000, 30_000, 100_000, 300_000, 1_000_000):
        vals = {"liftoff": [], "z": [], "sigma_star": []}
        for rep in range(20):
            pc = pair_cos(X, P=P, seed=rep)
            lift, *_ = occ.liftoff_cos(pc, D_BENCH, seed=rep)
            _, z = occ.stolarsky_z(pc, D_BENCH, seed=rep)
            vals["liftoff"].append(lift if lift is not None else np.nan)
            vals["z"].append(z)
            vals["sigma_star"].append(occ.sigma_star(pc, N_BENCH))
        rows.append({"P": P, **{k: {"mean": float(np.nanmean(v)),
                                    "sd": float(np.nanstd(v))}
                                for k, v in vals.items()}})
    json.dump(rows, open(out, "w"), indent=1)


# ---------------------------------------------------------------- E1c
def e1c(out):
    """DTM mass-fraction heat map over (m, pocket size k)."""
    rows = []
    for k in (20, 50, 200, 800):
        X = make_corpus(N_BENCH, D_BENCH, k, seed=1)
        pocket_idx = np.arange(N_BENCH - k, N_BENCH)
        for m in (0.005, 0.01, 0.02, 0.05, 0.1):
            f = cr.dtm(X, m_frac=m)
            sep = float(np.median(f[:N_BENCH - k]) / np.median(f[pocket_idx]))
            rows.append({"k": k, "m": m, "separation": sep,
                         "blur_boundary_m": k / N_BENCH})
    json.dump(rows, open(out, "w"), indent=1)


# ---------------------------------------------------------------- E1d
def e1d(out):
    """Merge-tree parameter sweep: planted-pocket recovery over the grid."""
    rows = []
    for count, size in ((1, 20), (1, 50), (1, 200), (5, 50)):
        rng = np.random.default_rng(2)
        n, d = 2000, 256
        X = unit(rng.standard_normal((n - count * size, d)))
        planted = []
        for c_i in range(count):
            c = unit(rng.standard_normal((1, d)))
            P = unit(c + 0.01 * rng.standard_normal((size, d)))
            planted.append(set(range(len(X), len(X) + size)))
            X = np.vstack([X, P])
        X = X.astype(np.float32)
        for min_size in (4, 8, 16, 32):
            pk, _ = cr.pockets(X, min_size=min_size, seed=0)
            hits = 0
            for pl in planted:
                for p in pk:
                    mem = set(int(m) for m in p["members"])
                    inter = len(mem & pl)
                    if inter >= 0.5 * len(pl) and inter >= 0.5 * len(mem):
                        hits += 1
                        break
            rows.append({"count": count, "size": size, "min_size": min_size,
                         "recovered": hits, "reported": len(pk)})
    json.dump(rows, open(out, "w"), indent=1)


# ---------------------------------------------------------------- E1e
def _e1e_one(seed):
    Xc = make_corpus(N_BENCH, D_BENCH, K_BENCH, seed=seed)
    Xu = make_corpus(N_BENCH, D_BENCH, 0, seed=seed)
    hc = headline(Xc, D_BENCH, seed=seed)
    hu = headline(Xu, D_BENCH, seed=seed)
    return {"seed": seed, "clumped": hc, "uniform": hu}


def e1e(out, procs):
    with Pool(procs) as p:
        rows = p.map(_e1e_one, range(20))
    json.dump(rows, open(out, "w"), indent=1)


# ---------------------------------------------------------------- E4c
def e4c(out):
    """Uncertainty of the Stolarsky z as a function of null-replicate count."""
    X = make_corpus(N_BENCH, D_BENCH, K_BENCH, seed=0)
    pc = pair_cos(X, seed=0)
    rows = []
    for reps in (8, 16, 24, 48, 96):
        zs = [occ.stolarsky_z(pc, D_BENCH, reps=reps, seed=100 + t)[1]
              for t in range(30)]
        rows.append({"null_reps": reps, "z_mean": float(np.mean(zs)),
                     "z_sd": float(np.std(zs))})
    json.dump(rows, open(out, "w"), indent=1)


# ---------------------------------------------------------------- E5
def _c_structured(X, Sigma_fn, sigma2_total, P=30_000, seed=0):
    """C at matched total noise power under a structured Gaussian channel:
    exact per pair, p = Phi(-(r^2/2)/sqrt(u' Sigma u)), u = y - x."""
    rng = np.random.default_rng(seed)
    n = len(X)
    i, j = rng.integers(0, n, P), rng.integers(0, n, P)
    keep = i != j
    U = X[i[keep]].astype(np.float64) - X[j[keep]].astype(np.float64)
    r2 = (U * U).sum(1)
    var = Sigma_fn(U)                          # u' Sigma u per row
    p = occ._phi(-(r2 / 2.0) / np.sqrt(np.maximum(var, 1e-30)))
    return float((n - 1) * p.mean())


def e5(out):
    """sigma* / C ranking stability under structured query-noise models."""
    d, n = 256, 2000
    gens = {
        "uniform": lambda r: unit(r.standard_normal((n, d))),
        "clump50": lambda r: np.vstack([unit(r.standard_normal((n - 50, d))),
                                        unit(unit(r.standard_normal((1, d)))
                                             + 0.05 * r.standard_normal((50, d)))]),
        "clump200": lambda r: np.vstack([unit(r.standard_normal((n - 200, d))),
                                         unit(unit(r.standard_normal((1, d)))
                                              + 0.05 * r.standard_normal((200, d)))]),
        "clumps5x100": lambda r: np.vstack(
            [unit(r.standard_normal((n - 500, d)))] +
            [unit(unit(r.standard_normal((1, d)))
                  + 0.05 * r.standard_normal((100, d))) for _ in range(5)]),
        "cone": lambda r: unit(r.standard_normal((n, d)) + 6 * np.eye(d)[0]),
        "lowrank": lambda r: unit(np.pad(r.standard_normal((n, 8)),
                                         ((0, 0), (0, d - 8)))),
        "cone+dup": lambda r: np.vstack(
            [unit(r.standard_normal((n - 200, d)) + 6 * np.eye(d)[0]),
             unit(unit(r.standard_normal((1, d)) + 6 * np.eye(d)[0])
                  + 0.02 * r.standard_normal((200, d)))]),
    }
    sigma = 0.25
    s2 = sigma * sigma
    results = {}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        corpora = {k: g(np.random.default_rng(seed * 10 + h)).astype(np.float32)
                   for h, (k, g) in enumerate(gens.items())}
        # noise models at matched total power s2 * d
        rot = np.linalg.qr(rng.standard_normal((d, d)))[0]
        for name, fn in {
            "iso": (lambda U: s2 * (U * U).sum(1)),
            "lowrank2": (lambda U: (s2 * d / 2) *
                         ((U @ rot[:, :2]) ** 2).sum(1)),
            "lowrank25": (lambda U: (s2 * d / 25) *
                          ((U @ rot[:, :25]) ** 2).sum(1)),
        }.items():
            key = f"{name}/seed{seed}"
            results[key] = {k: _c_structured(X, fn, s2, seed=seed)
                            for k, X in corpora.items()}
        # PC-aligned: noise along each corpus's own top-10 directions
        pcs = {}
        for k, X in corpora.items():
            lam, V = np.linalg.eigh(np.cov(X.T.astype(np.float64)))
            top = V[:, -10:]
            pcs[k] = _c_structured(
                X, lambda U, T=top: (s2 * d / 10) * ((U @ T) ** 2).sum(1),
                s2, seed=seed)
        results[f"pcalign/seed{seed}"] = pcs
    # rank stability vs iso, averaged over seeds
    from itertools import combinations
    names = list(gens)
    stab = {}
    for model in ("lowrank2", "lowrank25", "pcalign"):
        agr = []
        for seed in range(5):
            a = results[f"iso/seed{seed}"]
            b = results[f"{model}/seed{seed}"]
            pairs = list(combinations(names, 2))
            agr.append(np.mean([(a[x] - a[y]) * (b[x] - b[y]) > 0
                                for x, y in pairs]))
        stab[model] = {"order_agreement_mean": float(np.mean(agr)),
                       "sd": float(np.std(agr))}
    json.dump({"C_values": results, "rank_stability_vs_iso": stab},
              open(out, "w"), indent=1)


EXPERIMENTS = {"e1a": e1a, "e1b": e1b, "e1c": e1c, "e1d": e1d,
               "e4c": e4c, "e5": e5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None)
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 4))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = [a.only] if a.only else ["e1e"] + list(EXPERIMENTS)
    for name in todo:
        t0 = time.time()
        path = os.path.join(a.out, f"{name}.json")
        if name == "e1e":
            e1e(path, a.procs)
        else:
            EXPERIMENTS[name](path)
        print(f"{name}: done in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
