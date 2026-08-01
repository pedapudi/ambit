#!/usr/bin/env python3
"""Measure one round of the E6 loop: embeddings in, JSON readouts out.

Given the round's vectors (npy aligned to the subset manifest, or a parquet dir
whose uuids are matched to the manifest), computes the occupancy/crowding
readouts on the full subset and the comparison verdicts on the held-out slice:

  sigma_star (+ uniform-null ratio), liftoff cosine + rank-envelope global p,
  Stolarsky z, DTM low-tail (p1, median) vs the uniform band, pockets above the
  null prominence floor, per-item collision-count summary (incl. the flagged
  cohort = top 1% by BASE collisions, tracked across rounds), and held-out
  neighbor overlap@10 against the base model.

Held-out discipline: overlap and the flagged-cohort verdicts use ONLY held-out
rows; the training rows never grade themselves.

  python measure_round.py --subset subset200k --emb round1.npy \
      --round 1 --out rounds/round1.json
"""
import argparse, glob, json, os

import numpy as np

from ambit import crowding as cr
from ambit import knn, metrics
from ambit import occupancy as occ
from ambit.compare import neighbor_overlap

N_PAIRS = 200_000
DTM_CAP = 6_000
TREE_CAP = 4_096
OVERLAP_K = 10
FLAG_FRAC = 0.01


def load_vectors(path, manifest):
    if path.endswith(".npy"):
        X = np.load(path).astype(np.float32)
    else:
        import pyarrow.parquet as pq
        want = {u: i for i, u in enumerate(manifest["uuid"])}
        X = np.zeros((len(want), 0), np.float32)
        rows = []
        for f in sorted(glob.glob(os.path.join(path, "*.parquet"))):
            t = pq.read_table(f, columns=["uuid", "embedding"])
            us = t.column("uuid").to_pylist()
            v = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
            v = v.astype(np.float32).reshape(len(us), -1)
            for u, row in zip(us, v):
                i = want.get(u)
                if i is not None:
                    rows.append((i, row))
        if len(rows) != len(want):
            raise SystemExit(f"parquet covers {len(rows)}/{len(want)} manifest uuids")
        X = np.zeros((len(want), rows[0][1].shape[0]), np.float32)
        for i, row in rows:
            X[i] = row
    if len(X) != len(manifest["uuid"]):
        raise SystemExit(f"vector count {len(X)} != manifest n {len(manifest['uuid'])}")
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="dir from make_subset.py")
    ap.add_argument("--emb", required=True, help="round vectors: .npy or parquet dir")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-n", type=int, default=None,
                    help="n for sigma_star (default: subset size; set to the full "
                         "corpus size for TR-parity final runs)")
    ap.add_argument("--sigma-collisions", type=float, default=None,
                    help="fixed sigma for collision-count tracking across rounds "
                         "(default: this round's sigma_star)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.subset, "manifest.json")))
    X = load_vectors(a.emb, manifest)
    B = np.load(os.path.join(a.subset, "base.npy")).astype(np.float32)
    hold = np.asarray(manifest["split"], object) == "heldout"
    n_corpus = a.corpus_n or len(X)
    dim = X.shape[1]
    rng = np.random.default_rng(a.seed)
    out = {"round": a.round, "n": len(X), "dim": dim, "corpus_n_for_sigma": n_corpus}

    # ---- continuous layer (one shared pair sample) ----------------------
    pc = metrics.random_pair_cosine(X, n_pairs=N_PAIRS, normalized=True, seed=a.seed)
    out["mean_pair_cos"] = float(pc.mean())
    p_glob, lift, *_ = occ.rank_envelope(pc, dim, reps=99, seed=a.seed)
    out["liftoff_cos"] = float(lift) if lift is not None else None
    out["envelope_p"] = float(p_glob)
    _, z = occ.stolarsky_z(pc, dim, seed=a.seed)
    out["stolarsky_z"] = float(z)
    s_star = occ.sigma_star(pc, n_corpus)
    u_pc = occ.null_pair_cos(dim, N_PAIRS, seed=a.seed)
    out["sigma_star"] = float(s_star)
    out["sigma_star_uniform"] = float(occ.sigma_star(u_pc, n_corpus))

    # ---- structural layer ----------------------------------------------
    sub = X if len(X) <= DTM_CAP else X[rng.choice(len(X), DTM_CAP, replace=False)]
    d = cr.dtm(sub, m_frac=0.02)
    u_p1, _ = cr.dtm_null_band(len(sub), dim, m_frac=0.02, seed=a.seed)
    out["dtm_p1"] = float(np.quantile(d, 0.01))
    out["dtm_median"] = float(np.median(d))
    out["dtm_uniform_p1"] = float(u_p1)
    pk, _ = cr.pockets(X, min_size=8, max_n=TREE_CAP, max_pockets=10, seed=a.seed)
    floor = cr.null_prominence(min(len(X), TREE_CAP), dim, min_size=8, seed=a.seed)
    pk = [p for p in pk if p["prominence"] > 2 * floor]
    out["pocket_floor2x"] = float(2 * floor)
    out["pockets"] = [{"size": int(p["size"]), "birth": float(p["birth"]),
                       "death": float(p["death"]), "prominence": float(p["prominence"])}
                      for p in pk]

    # ---- per-item collisions, flagged cohort tracked from BASE ----------
    sigma_c = a.sigma_collisions or float(s_star)
    out["sigma_collisions"] = sigma_c
    cc = occ.collision_counts(X, sigma_c)
    cb = occ.collision_counts(B, sigma_c)
    k_flag = max(1, int(FLAG_FRAC * len(X)))
    flagged = np.argpartition(-cb, k_flag - 1)[:k_flag]        # base-defined cohort
    fh = flagged[hold[flagged]]                                # held-out members only
    out["collisions"] = {
        "median_all": float(np.median(cc)),
        "flagged_cohort_median_base": float(np.median(cb[fh])) if len(fh) else None,
        "flagged_cohort_median_now": float(np.median(cc[fh])) if len(fh) else None,
        "flagged_heldout_n": int(len(fh)),
    }

    # ---- held-out comparison vs base -----------------------------------
    Xh, Bh = X[hold], B[hold]
    ov = neighbor_overlap(knn.topk_cosine(Bh, OVERLAP_K), knn.topk_cosine(Xh, OVERLAP_K))
    out["heldout_neighbor_overlap@10"] = float(np.mean(ov))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
