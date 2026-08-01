"""Tier-2 validation: E2 (detection power vs the simple-baseline suite at
matched 1% false-alarm) and E4b (whole-instrument null calibration, including
the null distribution of pocket prominence).

All detectors are scalar test statistics with one-sided alarm directions;
every threshold is calibrated on dedicated pure-null replicates at the same
(n, d) and sample sizes, so the comparison is power against power at matched
specificity. The rank-envelope detector is self-calibrating (its own p).

Run: python experiments/tier2_power.py --out results
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from ambit import crowding as cr
from ambit import metrics
from ambit import occupancy as occ

N, D, P_PAIRS = 2000, 256, 60_000
FA_TARGET = 0.01
NULL_REPS = 400
CELL_REPS = 50


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def gen(kind, seed):
    rng = np.random.default_rng(seed)
    X = unit(rng.standard_normal((N, D)))
    if kind == "null":
        return X
    if kind == "cone":
        return unit(rng.standard_normal((N, D)) + 3 * np.eye(D)[0])
    if kind == "lowrank":
        return unit(np.pad(rng.standard_normal((N, 8)), ((0, 0), (0, D - 8))))
    count, size, spread = kind                     # pocket grid cell
    off = 0
    for _ in range(count):
        c = unit(rng.standard_normal((1, D)))
        X[off:off + size] = unit(c + spread * rng.standard_normal((size, D)))
        off += size
    return X


def pair_cos(X, seed):
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, N, P_PAIRS), rng.integers(0, N, P_PAIRS)
    keep = i != j
    return np.einsum("ij,ij->i", X[i[keep]].astype(np.float64),
                     X[j[keep]].astype(np.float64))


def detectors(X, seed):
    """All scalar statistics for one corpus. Alarm directions are applied at
    threshold time; here we just measure."""
    pc = pair_cos(X, seed)
    eigs = metrics.cov_eigs(X.astype(np.float64))
    G = X @ X.T
    np.fill_diagonal(G, -2)
    top = np.argpartition(-G, 10, axis=1)[:, :10]
    rows = np.arange(N)[:, None]
    knn_cos = np.take_along_axis(G, top, axis=1)
    occ_counts = np.bincount(top.ravel(), minlength=N).astype(float)
    hub = float(((occ_counts - occ_counts.mean()) ** 3).mean()
                / (occ_counts.std() or 1) ** 3)
    f = cr.dtm(X, m_frac=0.02)
    pk, _ = cr.pockets(X, min_size=8, seed=0)
    p_env, lift, *_ = occ.rank_envelope(pc, D, reps=99, seed=seed + 5)
    _, z = occ.stolarsky_z(pc, D, seed=seed)
    return {
        "mean_cos": float(pc.mean()),                       # alarm: high
        "isoscore": float(metrics.isoscore(eigs)),          # alarm: low
        "effrank": float(metrics.effective_rank(eigs)),     # alarm: low
        "uniformity": float(metrics.uniformity_from_cos(pc)),   # alarm: high
        "hubness": hub,                                     # alarm: high
        "knn_q05": float(np.percentile(1 - knn_cos.mean(1), 5)),  # alarm: low
        "z": z,                                             # alarm: low (negative)
        "dtm_ratio": float(np.percentile(f, 1) / np.median(f)),   # alarm: low
        "pocket_prom": float(max((p["prominence"] for p in pk), default=0.0)),
        "sigma_star": float(occ.sigma_star(pc, N)),         # alarm: low
        "env_p": float(p_env),                              # self-calibrating
        "env_lift_fired": bool(lift is not None),
    }


ALARM_LOW = {"isoscore", "effrank", "knn_q05", "z", "dtm_ratio", "sigma_star"}
ALARM_HIGH = {"mean_cos", "uniformity", "hubness", "pocket_prom"}


def _null_one(seed):
    return detectors(gen("null", 10_000 + seed), 10_000 + seed)


def _cell_one(args):
    kind, rep = args
    seed = abs(hash((str(kind), rep))) % (2**31)
    return kind, detectors(gen(kind, seed), seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 4))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    with Pool(a.procs) as pool:
        nulls = pool.map(_null_one, range(NULL_REPS))
    print(f"nulls: {NULL_REPS} in {time.time()-t0:.0f}s", flush=True)

    # ---- E4b: null calibration ------------------------------------------
    e4b = {
        "env_reject_rate_p01": float(np.mean([r["env_p"] <= 0.01 for r in nulls])),
        "env_lift_rate": float(np.mean([r["env_lift_fired"] for r in nulls])),
        "abs_z_gt3_rate": float(np.mean([abs(r["z"]) > 3 for r in nulls])),
        "pocket_prom_null": {
            "p50": float(np.percentile([r["pocket_prom"] for r in nulls], 50)),
            "p95": float(np.percentile([r["pocket_prom"] for r in nulls], 95)),
            "p99": float(np.percentile([r["pocket_prom"] for r in nulls], 99)),
        },
    }
    json.dump(e4b, open(os.path.join(a.out, "e4b.json"), "w"), indent=1)
    print("e4b:", e4b, flush=True)

    # ---- thresholds at matched false alarm ------------------------------
    thr = {}
    for k in ALARM_LOW:
        thr[k] = float(np.percentile([r[k] for r in nulls], 100 * FA_TARGET))
    for k in ALARM_HIGH:
        thr[k] = float(np.percentile([r[k] for r in nulls], 100 * (1 - FA_TARGET)))

    def alarms(r):
        out = {k: r[k] <= thr[k] for k in ALARM_LOW}
        out.update({k: r[k] >= thr[k] for k in ALARM_HIGH})
        out["env_p"] = r["env_p"] <= 0.01
        return out

    # measured false-alarm rates (should be ~FA_TARGET by construction)
    fa = {k: float(np.mean([alarms(r)[k] for r in nulls]))
          for k in list(ALARM_LOW | ALARM_HIGH) + ["env_p"]}

    # ---- E2: the pathology grid -----------------------------------------
    kinds = [(c, s, sp) for c in (1, 5) for s in (20, 50, 200)
             for sp in (0.005, 0.02, 0.06)] + ["cone", "lowrank"]
    jobs = [(k, r) for k in kinds for r in range(CELL_REPS)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(_cell_one, jobs)
    print(f"cells: {len(jobs)} in {time.time()-t0:.0f}s", flush=True)

    power = {}
    for kind in kinds:
        rs = [r for k, r in rows if k == kind]
        power[str(kind)] = {det: float(np.mean([alarms(r)[det] for r in rs]))
                            for det in fa}
    json.dump({"thresholds": thr, "false_alarm_measured": fa, "power": power},
              open(os.path.join(a.out, "e2.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
