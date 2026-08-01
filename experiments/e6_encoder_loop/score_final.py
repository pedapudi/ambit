#!/usr/bin/env python3
"""Blind-then-score: relate ambit's per-document readout to realized retrieval
failures, then check the tuned model fixes the predicted cohort. (H2 preview.)

Judged labels enter here and only here — as outcomes to score against, never as
inputs to any measurement or training step upstream.

Inputs: the per-query rank files written by run_eval.py (baseline_ranks.json
format) for the base and tuned models, the base eCFR corpus vectors (the eval
cache .npy), and the corpus/questions jsonl for uuid -> row alignment.

Reports:
  1. participation-in-failure AUC — do BASE per-document expected-collision
     counts predict which judged-relevant documents fail at their own queries
     (rank > --fail-rank)?  Control: top-1 neighbor cosine (raw proximity).
  2. cohort deltas — rank and failure-rate changes base -> tuned, split by the
     predicted high-collision cohort vs the rest.

  python score_final.py --corpus-npy eval/cache/ecfr_corpus.npy \
      --corpus-jsonl eval/ecfr_corpus.jsonl --questions eval/questions.jsonl \
      --ranks-base eval/baseline_ranks.json --ranks-tuned eval/tuned_ranks.json \
      --sigma 0.1226 --out scores.json
"""
import argparse, json

import numpy as np

from ambit import crowding as cr
from ambit import occupancy as occ


def auc(score, label):
    """Mann-Whitney AUC of `score` predicting boolean `label` (ties = 0.5)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # midranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-npy", required=True, help="BASE eCFR corpus vectors")
    ap.add_argument("--corpus-jsonl", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--ranks-base", required=True)
    ap.add_argument("--ranks-tuned", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--fail-rank", type=int, default=10)
    ap.add_argument("--cohort-frac", type=float, default=0.1)
    ap.add_argument("--out", default="scores.json")
    a = ap.parse_args()

    X = np.load(a.corpus_npy).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    docs = [json.loads(l) for l in open(a.corpus_jsonl)]
    docpos = {d["doc_uuid"]: i for i, d in enumerate(docs)}
    qs = {q["qid"]: q for q in (json.loads(l) for l in open(a.questions))}
    rb = {r["qid"]: r["rank"] for r in json.load(open(a.ranks_base))}
    rt = {r["qid"]: r["rank"] for r in json.load(open(a.ranks_tuned))}
    common = [qid for qid in rb if qid in rt and qid in qs]
    print(f"{len(common)} scored queries | corpus {X.shape}", flush=True)

    # per-document predictors from the BASE embedding only
    cc = occ.collision_counts(X, a.sigma)
    top1 = cr.topk_cos_values(X, 1)[:, 0]

    # doc-level outcome: participates in a failure at any of its own queries
    doc_q = {}
    for qid in common:
        doc_q.setdefault(docpos[qs[qid]["gold_uuid"]], []).append(qid)
    d_idx = np.array(sorted(doc_q))
    fail_b = np.array([any(rb[q] > a.fail_rank for q in doc_q[i]) for i in d_idx])
    fail_t = np.array([any(rt[q] > a.fail_rank for q in doc_q[i]) for i in d_idx])

    out = {
        "n_queries": len(common), "n_gold_docs": len(d_idx),
        "fail_rank": a.fail_rank, "sigma": a.sigma,
        "auc_collisions": auc(cc[d_idx], fail_b),
        "auc_top1cos_control": auc(top1[d_idx], fail_b),
    }

    # cohort deltas: predicted high-collision decile vs the rest
    k = max(1, int(a.cohort_frac * len(d_idx)))
    hot = np.zeros(len(d_idx), bool)
    hot[np.argpartition(-cc[d_idx], k - 1)[:k]] = True
    for name, m in (("high_collision_cohort", hot), ("rest", ~hot)):
        qids = [q for i in d_idx[m] for q in doc_q[i]]
        b = np.array([rb[q] for q in qids]); t = np.array([rt[q] for q in qids])
        out[name] = {
            "n_docs": int(m.sum()), "n_queries": len(qids),
            "median_rank_base": float(np.median(b)),
            "median_rank_tuned": float(np.median(t)),
            "fail_rate_base": float(np.mean(b > a.fail_rank)),
            "fail_rate_tuned": float(np.mean(t > a.fail_rank)),
            "doc_fail_rate_base": float(fail_b[m].mean()),
            "doc_fail_rate_tuned": float(fail_t[m].mean()),
        }

    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
