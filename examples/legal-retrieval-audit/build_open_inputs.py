#!/usr/bin/env python3
"""Build the authoring inputs for the open-corpus eval:
  single.jsonl            2500 single docs           -> 1 question each, gold={uuid}
  clusters_sibling.jsonl  1250 same-Part eCFR pairs  -> 1 multi-hop Q, gold={uuids}
  clusters_semantic.jsonl 1250 embedding-neighbor sets-> 1 multi-hop Q, gold={uuids}
All members carry text so LLM subagents can author grounded questions offline.
Embeddings are looked up from the already-computed 1M map + eCFR cache."""
import glob, json, os, random, re
import numpy as np, pyarrow.parquet as pq

EVAL_OPEN = "eval-open"
MAP   = "emb-1024-1M"
ECFR_NPY  = "eval/cache/ecfr_corpus.npy"
ECFR_META = "eval/cache/ecfr_corpus.meta.json"
ECFR_CORPUS = "eval/ecfr_corpus.jsonl"
rng = random.Random(0)

cands = [json.loads(l) for l in open(os.path.join(EVAL_OPEN, "candidates.jsonl"))]
by_uuid = {c["uuid"]: c for c in cands}
print(f"candidates: {len(cands)}")

# ---- single ----
single = cands[:2500]
with open(os.path.join(EVAL_OPEN, "single.jsonl"), "w") as fo:
    for c in single: fo.write(json.dumps(c) + "\n")
print(f"single: {len(single)}")

# ---- embeddings for candidates (eCFR from cache, others from the 1M map) ----
emb = {}
ids = json.load(open(ECFR_META))["ids"]; V = np.load(ECFR_NPY)
pos = {u: i for i, u in enumerate(ids)}
for c in cands:
    if c["subset"] == "eCFR" and c["uuid"] in pos:
        emb[c["uuid"]] = V[pos[c["uuid"]]]
want = {c["uuid"] for c in cands if c["uuid"] not in emb}
for f in sorted(glob.glob(os.path.join(MAP, "*.parquet"))):
    t = pq.read_table(f, columns=["uuid", "embedding"])
    us = t.column("uuid").to_pylist()
    M = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)
    for u, v in zip(us, M):
        if u in want: emb[u] = v
    if want.issubset(emb.keys()): break
print(f"candidate embeddings resolved: {len(emb)}/{len(cands)}")

# ---- semantic clusters (anchor + 2 nearest candidate neighbors, cross-domain ok) ----
ce = [c for c in cands if c["uuid"] in emb]
X = np.stack([emb[c["uuid"]] for c in ce]); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
order = list(range(len(ce))); rng.shuffle(order)
used, sem = set(), []
for ai in order:
    if len(sem) >= 1250: break
    if ai in used: continue
    sims = X @ X[ai]; sims[ai] = -2
    nbrs = [j for j in np.argsort(-sims)[:6] if j not in used][:2]
    if len(nbrs) < 2: continue
    members = [ce[ai]] + [ce[j] for j in nbrs]
    used.update([ai] + nbrs)
    sem.append({"cluster_id": f"sem-{len(sem):04d}", "type": "semantic",
                "members": [{"uuid": m["uuid"], "subset": m["subset"], "text": m["text"]} for m in members]})
with open(os.path.join(EVAL_OPEN, "clusters_semantic.jsonl"), "w") as fo:
    for c in sem: fo.write(json.dumps(c) + "\n")
print(f"semantic clusters: {len(sem)}")

# ---- sibling clusters (2 distinct eCFR docs sharing a Title+Part) ----
ecorp = [json.loads(l) for l in open(ECFR_CORPUS)]
groups = {}
for d in ecorp:
    parts = {s.split(".")[0] for s in d.get("sections", [])}
    for p in parts:
        groups.setdefault((d["title"], p), []).append(d)
keys = [k for k, v in groups.items() if len(v) >= 2]; rng.shuffle(keys)
sib = []
for k in keys:
    if len(sib) >= 1250: break
    docs = groups[k]; rng.shuffle(docs)
    m = docs[:2]
    sib.append({"cluster_id": f"sib-{len(sib):04d}", "type": "sibling", "title": k[0], "part": k[1],
                "members": [{"uuid": d["doc_uuid"], "subset": "eCFR", "text": d["text"][:1500]} for d in m]})
with open(os.path.join(EVAL_OPEN, "clusters_sibling.jsonl"), "w") as fo:
    for c in sib: fo.write(json.dumps(c) + "\n")
print(f"sibling clusters: {len(sib)}")
