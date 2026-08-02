#!/usr/bin/env python3
"""Case-study-parity measurement of a full embedding map (parquet dir).

Streams the map with ambit.scan (reservoir 20k, the frozen default), then
computes the readouts of the technical report's case-study table: global
suite (mean pair cosine, IsoScore, effective rank, hubness skew), continuous
layer (rank-envelope p + liftoff, occupancy z, sigma* at corpus n), per-item
layer (DTM low tail vs uniform), and structure (pockets above the null
prominence floor).

  python measure_map.py --map ~/e6-map-1M --out map1M.json
"""
import argparse, json

import numpy as np

import ambit
from ambit import crowding as cr
from ambit import knn, metrics
from ambit import occupancy as occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sc = ambit.scan(a.map, embedding_col="embedding")
    es = sc.sample.normalize()
    X = np.asarray(es.X, np.float32)
    n, dim = int(sc.n), int(sc.dim)
    out = {"n": n, "dim": dim, "reservoir": len(X)}

    eigs = sc.eigs
    out["effective_rank"] = float(metrics.effective_rank(eigs))
    out["isoscore"] = float(metrics.isoscore(eigs))

    pc = metrics.random_pair_cosine(X, n_pairs=200_000, normalized=True, seed=a.seed)
    out["mean_pair_cos"] = float(pc.mean())
    p_glob, lift, *_ = occ.rank_envelope(pc, dim, reps=99, seed=a.seed)
    out["envelope_p"] = float(p_glob)
    out["liftoff_cos"] = float(lift) if lift is not None else None
    _, z = occ.stolarsky_z(pc, dim, seed=a.seed)
    out["stolarsky_z"] = float(z)
    out["sigma_star"] = float(occ.sigma_star(pc, n))
    out["sigma_star_uniform"] = float(
        occ.sigma_star(occ.null_pair_cos(dim, 200_000, seed=a.seed), n))

    idx = knn.topk_cosine(X, 10)
    out["hubness_skew"] = float(metrics.hubness_skew(idx))

    rng = np.random.default_rng(a.seed)
    sub = X if len(X) <= 6000 else X[rng.choice(len(X), 6000, replace=False)]
    d = cr.dtm(sub, m_frac=0.02)
    u_p1, _ = cr.dtm_null_band(len(sub), dim, m_frac=0.02, seed=a.seed)
    out["dtm_p1"] = float(np.quantile(d, 0.01))
    out["dtm_median"] = float(np.median(d))
    out["dtm_uniform_p1"] = float(u_p1)

    pk, _ = cr.pockets(X, min_size=8, max_n=4096, max_pockets=10, seed=a.seed)
    floor = cr.null_prominence(min(len(X), 4096), dim, min_size=8, seed=a.seed)
    out["pocket_floor2x"] = float(2 * floor)
    out["pockets"] = [{"size": int(p["size"]), "birth": float(p["birth"]),
                       "death": float(p["death"]),
                       "prominence": float(p["prominence"])}
                      for p in pk if p["prominence"] > 2 * floor]

    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
