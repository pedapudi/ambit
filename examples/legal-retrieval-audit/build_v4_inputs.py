#!/usr/bin/env python3
"""v4 — build a FRESH batch of MULTI-DOC training clusters, DISJOINT from the eval.

Same shapes as build_open_inputs.py (sibling = same Title+Part eCFR pair; semantic =
embedding neighbors) but: a new seed, a fresh candidate pool, a higher cap, and — the
key addition — it EXCLUDES any cluster whose member uuids touch an eval gold doc
(Set A eval gold sections + Set B eval gold_uuids). That guarantees the authored v4
TRAINING questions cannot be near-paraphrases of eval multi-hop questions (the same
cluster-disjointness Set B's train/eval split had by construction). All -> train.

Run after a fresh candidate pool, e.g.:
  python build_open_candidates.py --seed 42 --per-subset 4000 --out eval-open/candidates_v4.jsonl
  python build_v4_inputs.py
Output: eval-open/clusters_v4.jsonl  (sibling + semantic)
"""
import glob, json, os, random
import numpy as np, pyarrow.parquet as pq

MAP = "emb-1024-1M"
ECFR_NPY = "eval/cache/ecfr_corpus.npy"
ECFR_META = "eval/cache/ecfr_corpus.meta.json"
ECFR_CORPUS = "eval/ecfr_corpus.jsonl"
CANDS = "eval-open/candidates_v4.jsonl"
EVAL_A = "eval-data/questions.jsonl"          # Set A (gold_uuid + split)
EVAL_B = "eval-data/questions_open.jsonl"     # Set B (gold_uuids + split)
OUT = "eval-open/clusters_v4.jsonl"
SIB_CAP, SEM_CAP = 2200, 2800
rng = random.Random(42)

# ---- eval gold uuids to exclude (Set A eval gold sections + Set B eval gold_uuids) ----
evg = set()
for l in open(EVAL_A):
    r = json.loads(l)
    if r.get("split") == "eval":
        evg.add(r["gold_uuid"])
if os.path.exists(EVAL_B):
    for l in open(EVAL_B):
        r = json.loads(l)
        if r.get("split") == "eval":
            evg.update(r["gold_uuids"])
print(f"eval gold uuids to exclude: {len(evg)}", flush=True)

cands = [json.loads(l) for l in open(CANDS)]
cands = [c for c in cands if c["uuid"] not in evg]          # drop eval-gold candidates outright
print(f"candidates (post eval-gold filter): {len(cands)}", flush=True)

# ---- candidate embeddings (eCFR from cache, others from the 1M map) ----
emb = {}
ids = json.load(open(ECFR_META))["ids"]; V = np.load(ECFR_NPY)
pos = {u: i for i, u in enumerate(ids)}
for c in cands:
    if c["subset"] == "eCFR" and c["uuid"] in pos:
        emb[c["uuid"]] = V[pos[c["uuid"]]]
want = {c["uuid"] for c in cands if c["uuid"] not in emb}
for f in sorted(glob.glob(MAP + "/*.parquet")):
    t = pq.read_table(f, columns=["uuid", "embedding"]); us = t.column("uuid").to_pylist()
    M = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)
    for u, v in zip(us, M):
        if u in want:
            emb[u] = v
    if want.issubset(emb.keys()):
        break
print(f"candidate embeddings resolved: {len(emb)}/{len(cands)}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
out = []

# ---- semantic clusters (anchor + 2 nearest candidate neighbors, cross-domain ok) ----
ce = [c for c in cands if c["uuid"] in emb]
X = np.stack([emb[c["uuid"]] for c in ce]); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
order = list(range(len(ce))); rng.shuffle(order)
used, nsem = set(), 0
for ai in order:
    if nsem >= SEM_CAP:
        break
    if ai in used:
        continue
    sims = X @ X[ai]; sims[ai] = -2
    nbrs = [j for j in np.argsort(-sims)[:8] if j not in used][:2]
    if len(nbrs) < 2:
        continue
    members = [ce[ai]] + [ce[j] for j in nbrs]
    if any(m["uuid"] in evg for m in members):           # belt-and-suspenders
        continue
    used.update([ai] + nbrs)
    out.append({"cluster_id": f"v4sem-{nsem:04d}", "type": "semantic",
                "members": [{"uuid": m["uuid"], "subset": m["subset"], "text": m["text"]} for m in members]})
    nsem += 1
print(f"semantic clusters: {nsem}", flush=True)

# ---- sibling clusters (2 distinct eCFR docs sharing a Title+Part), excluding eval golds ----
ecorp = [json.loads(l) for l in open(ECFR_CORPUS)]
groups = {}
for d in ecorp:
    if d["doc_uuid"] in evg:
        continue
    parts = {s.split(".")[0] for s in d.get("sections", [])}
    for p in parts:
        groups.setdefault((d["title"], p), []).append(d)
keys = [k for k, v in groups.items() if len(v) >= 2]; rng.shuffle(keys)
nsib = 0
for k in keys:
    if nsib >= SIB_CAP:
        break
    docs = groups[k][:]; rng.shuffle(docs); m = docs[:2]
    out.append({"cluster_id": f"v4sib-{nsib:04d}", "type": "sibling", "title": k[0], "part": k[1],
                "members": [{"uuid": d["doc_uuid"], "subset": "eCFR", "text": d["text"][:1500]} for d in m]})
    nsib += 1
print(f"sibling clusters: {nsib}", flush=True)

rng.shuffle(out)
with open(OUT, "w") as fo:
    for c in out:
        fo.write(json.dumps(c) + "\n")
print(f"wrote {len(out)} clusters -> {OUT}  (sem {nsem} + sib {nsib})", flush=True)
