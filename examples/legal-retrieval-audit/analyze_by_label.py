#!/usr/bin/env python3
"""Per-label occupancy / retrieval-resolution analysis over an ambit-style parquet
embedding set (columns: an embedding column + a label column). Complements the
aggregate `ambit report` with a by-label breakdown:

  - global mean random-pair cosine (anisotropy) and kNN hubness skew
  - per label: intra-cohesion, kNN purity (does a doc's neighbors share its label?),
    and the nearest *other* label by centroid cosine
  - the full label x label centroid cosine matrix (which labels collapse together)

Usage:  python analyze_by_label.py <dir-of-parquet> [--label-col subset] [--embedding-col embedding]
"""
import argparse, glob
import numpy as np, pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew


def load(files, emb_col, lab_col, strip):
    Xs, labs = [], []
    for f in files:
        t = pq.read_table(f, columns=[emb_col, lab_col])
        col = t.column(emb_col).combine_chunks()
        v = col.values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)
        Xs.append(v)
        labs.extend((s.replace(strip, "") if strip else s) for s in t.column(lab_col).to_pylist())
    return np.vstack(Xs), np.array(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--embedding-col", default="embedding")
    ap.add_argument("--label-col", default="subset")
    ap.add_argument("--strip-prefix", default="")
    ap.add_argument("--per-label-sample", type=int, default=2500)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    X, labs = load(sorted(glob.glob(a.dir + "/*.parquet")), a.embedding_col, a.label_col, a.strip_prefix)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    labels = sorted(set(labs.tolist()))
    print(f"loaded {X.shape[0]:,} x {X.shape[1]} | {len(labels)} labels\n")

    i = rng.integers(0, len(X), 200_000); j = rng.integers(0, len(X), 200_000); m = i != j
    print(f"global mean random-pair cosine: {np.mean(np.sum(X[i[m]]*X[j[m]], axis=1)):+.3f}")

    cent = {s: X[labs == s].mean(0) for s in labels}
    for s in labels: cent[s] /= np.linalg.norm(cent[s]) + 1e-9
    cosmat = np.vstack([cent[s] for s in labels]) @ np.vstack([cent[s] for s in labels]).T

    def intra(s, cap=4000):
        idx = np.where(labs == s)[0]
        aa = rng.choice(idx, min(cap, len(idx))); bb = rng.choice(idx, min(cap, len(idx))); mm = aa != bb
        return float(np.mean(np.sum(X[aa[mm]] * X[bb[mm]], axis=1)))

    samp = np.concatenate([rng.choice(np.where(labs == s)[0], min(a.per_label_sample, int(np.sum(labs == s))),
                                      replace=False) for s in labels])
    Xs, ys = X[samp], labs[samp]
    _, ind = NearestNeighbors(n_neighbors=a.k + 1, metric="cosine").fit(Xs).kneighbors(Xs)
    neigh = ind[:, 1:]; purity = (ys[neigh] == ys[:, None]).mean(1)
    occ = np.bincount(neigh.ravel(), minlength=len(Xs))
    print(f"kNN@{a.k} hubness skew: {skew(occ):+.2f}\n")

    print(f"{'label':40s} {'n':>8s} {'intra':>6s} {'purity':>7s}  nearest-other (centroid cos)")
    print("-" * 100)
    for s in sorted(labels, key=lambda s: -np.mean(purity[ys == s])):
        si = labels.index(s); row = cosmat[si].copy(); row[si] = -1; k = int(np.argmax(row))
        print(f"{s:40s} {int(np.sum(labs==s)):8d} {intra(s):+6.2f} {float(np.mean(purity[ys==s])):7.2f}  {labels[k]:30s} {row[k]:+.2f}")
    print(f"\noverall kNN@{a.k} purity: {purity.mean():.2f}")
    for n, s in enumerate(labels): print(f"  [{n}] {s}")
    np.set_printoptions(precision=2, suppress=True, linewidth=200); print(cosmat)


if __name__ == "__main__":
    main()
