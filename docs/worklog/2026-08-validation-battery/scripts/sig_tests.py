import json, math
import numpy as np

import sys
sys.path.insert(0, "/home/sunil/ambit/src")
from ambit import occupancy as occ, metrics

rng = np.random.default_rng(0)
B = 10_000

# ---------- 1) retrieval: McNemar + paired bootstrap on MRR ----------
rb = {r["qid"]: r["rank"] for r in json.load(open("/home/sunil/e6-eval/ranks_base.json"))}
rt = {r["qid"]: r["rank"] for r in json.load(open("/home/sunil/e6-eval-r3/baseline_ranks.json"))}
q = sorted(set(rb) & set(rt))
b_r = np.array([rb[k] for k in q]); t_r = np.array([rt[k] for k in q])
print(f"queries: {len(q)}")
for K in (1, 10):
    sb, st = b_r <= K, t_r <= K
    b01 = int(np.sum(sb & ~st)); c10 = int(np.sum(~sb & st))
    n = b01 + c10
    p = sum(math.comb(n, i) for i in range(0, min(b01, c10) + 1)) / 2**n * 2 if n else 1.0
    print(f"R@{K}: base {sb.mean():.3f} tuned {st.mean():.3f} | discordant {b01} vs {c10} | McNemar exact p = {min(p,1.0):.4f}")
mrr = lambda r: np.where(r <= 10, 1.0 / r, 0.0)
d = mrr(t_r) - mrr(b_r)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
print(f"dMRR@10 = {d.mean():+.4f}, 95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]")

# ---------- 2) held-out flagged-cohort collisions: paired tests ----------
man = json.load(open("/home/sunil/e6-subset200k/manifest.json"))
hold = np.asarray(man["split"], object) == "heldout"
base = np.load("/home/sunil/e6-subset200k/base.npy").astype(np.float32)
sig = 0.13373944590126788
cb = occ.collision_counts(base, sig)
k = max(1, int(0.01 * len(base)))
flagged = np.argpartition(-cb, k - 1)[:k]
fh = flagged[hold[flagged]]
for name, path in (("LoRA r2", "/home/sunil/e6-rounds/r2.npy"), ("full FT", "/home/sunil/e6-rounds/r3.npy")):
    X = np.load(path).astype(np.float32)
    cc = occ.collision_counts(X, sig)
    dd = cc[fh] - cb[fh]
    neg = int((dd < 0).sum()); pos = int((dd > 0).sum()); n = neg + pos
    ps = sum(math.comb(n, i) for i in range(0, min(neg, pos) + 1)) / 2**n * 2
    bootm = np.array([np.median(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(B)])
    print(f"{name}: cohort n={len(fh)}, median delta {np.median(dd):+.4f}, "
          f"95% CI [{np.percentile(bootm,2.5):+.4f}, {np.percentile(bootm,97.5):+.4f}], "
          f"improved {neg}/{n}, sign-test p = {min(ps,1.0):.2e}")

# ---------- 3) AUC vs chance and vs control (paired bootstrap) ----------
import glob
sys.path.insert(0, "/home/sunil/ambit/experiments/e6_encoder_loop")
from score_final import auc
Xc = np.load("/home/sunil/e6-eval/cache/ecfr_corpus.npy").astype(np.float32)
Xc /= np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
docs = [json.loads(l) for l in open("/home/sunil/e6-eval/ecfr_corpus.jsonl")]
docpos = {d["doc_uuid"]: i for i, d in enumerate(docs)}
qs = {j["qid"]: j for j in (json.loads(l) for l in open("/home/sunil/e6-eval/questions.jsonl"))}
from ambit import crowding as cr
ccx = occ.collision_counts(Xc, 0.145)
top1 = cr.topk_cos_values(Xc, 1)[:, 0]
doc_q = {}
for qid in q:
    doc_q.setdefault(docpos[qs[qid]["gold_uuid"]], []).append(qid)
d_idx = np.array(sorted(doc_q))
fail = np.array([any(rb[x] > 10 for x in doc_q[i]) for i in d_idx])
s_coll, s_ctrl = ccx[d_idx], top1[d_idx]
a1, a2 = auc(s_coll, fail), auc(s_ctrl, fail)
bd = []
for _ in range(B):
    ii = rng.integers(0, len(d_idx), len(d_idx))
    if fail[ii].all() or (~fail[ii]).all(): continue
    bd.append((auc(s_coll[ii], fail[ii]), auc(s_ctrl[ii], fail[ii])))
bd = np.array(bd)
print(f"AUC collisions {a1:.3f}, 95% CI [{np.percentile(bd[:,0],2.5):.3f}, {np.percentile(bd[:,0],97.5):.3f}] (chance 0.5)")
dab = bd[:, 0] - bd[:, 1]
print(f"AUC diff vs control {a1-a2:+.3f}, 95% CI [{np.percentile(dab,2.5):+.3f}, {np.percentile(dab,97.5):+.3f}]")

# ---------- 4) sigma* differences vs pair-sampling noise ----------
r3 = np.load("/home/sunil/e6-rounds/r3.npy").astype(np.float32)
sb_, st_ = [], []
for s in range(8):
    sb_.append(occ.sigma_star(metrics.random_pair_cosine(base, n_pairs=200_000, normalized=True, seed=100+s), len(base)))
    st_.append(occ.sigma_star(metrics.random_pair_cosine(r3, n_pairs=200_000, normalized=True, seed=100+s), len(base)))
sb_, st_ = np.array(sb_), np.array(st_)
print(f"sigma* base {sb_.mean():.5f} sd {sb_.std():.5f} | full-FT {st_.mean():.5f} sd {st_.std():.5f} | paired mean delta {np.mean(st_-sb_):+.5f} sd {np.std(st_-sb_):.5f}")
