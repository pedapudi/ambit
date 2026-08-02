#!/usr/bin/env python3
"""Dedup routing experiment: does the measurement-prescribed data repair
produce the retrieval gain that training could not?

The E6 loop found the corpus's crowding is textual near-duplication and the
staged protocol routes that to data repair. This script executes the repair
label-blind and scores it:

  Phase 1 (geometry only, no labels): find near-duplicate groups — pairs of
  documents with cosine >= the measured liftoff scale — via exact blocked
  GPU kNN over the base 1M map; canonicalize each group to its
  lexicographically-first uuid (a rule that cannot peek at judgments).

  Phase 2 (scoring, labels enter only here): re-rank the frozen eval queries
  in two arms.
    A. distractor arm: open-corpus eval (eCFR corpus + non-eCFR map rows as
       distractors); after = duplicate distractors dropped.
    B. in-corpus arm: closed eval over the eCFR corpus; after = corpus
       deduped, with qrels mapped through the canonicalization (a judged
       doc's group inherits the judgment) — mechanical, label-blind rule.

  Reports group/drop counts, paired ranks before/after, McNemar on
  fail(rank>10), and median ranks.

GPU host.  PYTHONPATH=src python dedup_routing.py --map ~/maps/legal/base-1024-1M \
  --eval-dir ~/e6-eval --threshold 0.8165 --out ~/e6-rounds/dedup_routing.json
"""
import argparse, glob, json, math

import numpy as np


def load_map(map_dir):
    import pyarrow.parquet as pq
    us, subs, vs = [], [], []
    for f in sorted(glob.glob(map_dir + "/*.parquet")):
        t = pq.read_table(f, columns=["uuid", "subset", "embedding"])
        us.extend(t.column("uuid").to_pylist())
        subs.extend(t.column("subset").to_pylist())
        v = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        vs.append(v.astype(np.float32).reshape(len(t), -1))
    X = np.vstack(vs)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return us, np.asarray(subs), X


def dup_groups(X, uuids, thr, device, k=32, block=4096):
    """Union-find over pairs with cosine >= thr, exact blocked GPU kNN."""
    import torch
    Xt = torch.tensor(X, device=device)
    n = len(X)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n_edges = 0
    for s in range(0, n, block):
        e = min(s + block, n)
        S = Xt[s:e] @ Xt.T
        S[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -2.0
        vals, idx = torch.topk(S, k, dim=1)
        m = vals >= thr
        rows = torch.nonzero(m)
        for r, c in rows.cpu().numpy():
            a, b = find(s + r), find(int(idx[r, c]))
            if a != b:
                parent[max(a, b)] = min(a, b)
            n_edges += 1
    roots = np.array([find(i) for i in range(n)])
    # canonical = lexicographically-first uuid within each group
    canon = {}
    for grp in np.unique(roots):
        members = np.flatnonzero(roots == grp)
        if len(members) > 1:
            first = members[np.argmin([uuids[m] for m in members])]
            for m in members:
                canon[m] = int(first)
    return canon, n_edges


def ranks_against(Q, gold_vecs, blocks):
    cnt = np.zeros(len(Q), dtype=np.int64)
    for D in blocks:
        for i in range(0, len(D), 20000):
            cnt += (Q @ D[i:i + 20000].T > np.sum(Q * gold_vecs, 1)[:, None]).sum(1)
    return cnt + 1


def mcnemar(before, after, fail_rank=10):
    fb, fa = before > fail_rank, after > fail_rank
    b01, c10 = int(np.sum(fb & ~fa)), int(np.sum(~fb & fa))
    n = b01 + c10
    p = sum(math.comb(n, i) for i in range(0, min(b01, c10) + 1)) / 2**n * 2 if n else 1.0
    return {"fixed": b01, "broken": c10, "p": min(p, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--threshold", type=float, required=True,
                    help="measured liftoff cosine (near-dup scale)")
    ap.add_argument("--exclude-label", default="eCFR-QA")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = {"threshold": a.threshold}

    # ---- eval assets (vectors precomputed in the base caches) -----------
    qs_all = [json.loads(l) for l in open(a.eval_dir + "/questions.jsonl")]
    corpus = [json.loads(l) for l in open(a.eval_dir + "/ecfr_corpus.jsonl")]
    D = np.load(a.eval_dir + "/cache/ecfr_corpus.npy").astype(np.float32)
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    Qall = np.load(a.eval_dir + "/cache/queries.with-instruction.npy").astype(np.float32)
    Qall /= np.maximum(np.linalg.norm(Qall, axis=1, keepdims=True), 1e-12)
    sel = [i for i, q in enumerate(qs_all) if q.get("split") == a.split]
    qs = [qs_all[i] for i in sel]
    Q = Qall[sel]
    docpos = {c["doc_uuid"]: i for i, c in enumerate(corpus)}
    gold = np.array([docpos[q["gold_uuid"]] for q in qs])
    print(f"{len(qs)} queries | corpus {D.shape}", flush=True)

    # ---- Phase 1a: dup groups among NON-excluded map rows (distractors) --
    uuids, subs, X = load_map(a.map)
    keep_lbl = np.array([a.exclude_label not in s for s in subs])
    du, dX = [uuids[i] for i in np.flatnonzero(keep_lbl)], X[keep_lbl]
    canon_d, edges_d = dup_groups(dX, du, a.threshold, a.device)
    drop_d = np.array(sorted(canon_d.keys()))
    drop_mask = np.zeros(len(dX), bool)
    drop_mask[[i for i in drop_d if canon_d[i] != i]] = True
    out["distractors"] = {"n": len(dX), "edges": edges_d,
                          "in_groups": len(canon_d),
                          "dropped": int(drop_mask.sum())}
    print(f"distractors: {out['distractors']}", flush=True)

    # ---- Phase 1b: dup groups within the eCFR corpus ---------------------
    cu = [c["doc_uuid"] for c in corpus]
    canon_c, edges_c = dup_groups(D, cu, a.threshold, a.device)
    drop_c = {i for i in canon_c if canon_c[i] != i}
    out["corpus"] = {"n": len(D), "edges": edges_c, "in_groups": len(canon_c),
                     "dropped": len(drop_c)}
    print(f"corpus: {out['corpus']}", flush=True)

    # ---- Phase 2A: distractor arm ---------------------------------------
    gv = D[gold]
    before = ranks_against(Q, gv, [D, dX])
    after = ranks_against(Q, gv, [D, dX[~drop_mask]])
    out["arm_distractor"] = {
        "median_before": float(np.median(before)),
        "median_after": float(np.median(after)),
        "R@10_before": float(np.mean(before <= 10)),
        "R@10_after": float(np.mean(after <= 10)),
        "mcnemar": mcnemar(before, after)}
    print(f"arm A: {out['arm_distractor']}", flush=True)

    # ---- Phase 2B: in-corpus arm (qrels mapped through canonicalization) -
    canon_of = lambda i: canon_c.get(i, i)
    gold_after = np.array([canon_of(g) for g in gold])
    keep_c = np.ones(len(D), bool)
    keep_c[list(drop_c)] = False
    # rank of the canonical gold within the deduped corpus
    before_c = ranks_against(Q, D[gold], [D])
    after_c = ranks_against(Q, D[gold_after], [D[keep_c]])
    out["arm_corpus"] = {
        "golds_remapped": int(np.sum(gold_after != gold)),
        "median_before": float(np.median(before_c)),
        "median_after": float(np.median(after_c)),
        "R@10_before": float(np.mean(before_c <= 10)),
        "R@10_after": float(np.mean(after_c <= 10)),
        "mcnemar": mcnemar(before_c, after_c)}
    print(f"arm B: {out['arm_corpus']}", flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
