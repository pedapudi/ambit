#!/usr/bin/env python3
"""v3 training data — MULTI-POSITIVE (no flatten). Regroup pairs.jsonl by query so
every multi-gold question keeps ALL its golds as positives, then attach the hard
negatives already mined (excluding the full gold set) in train_ecfr.jsonl /
train_synth.jsonl (run mine_hard_negs.py + mine_synth_negs.py first).

Output train_v3.jsonl, fixed columns for DDP shape-safety:
  anchor, positive_1..3 (padding = copy of positive_1), negative_1..5,
  label = [n_pos, puid_1, puid_2, puid_3]   (puid = GLOBAL int id per gold uuid;
          padded positive cols get -1). The loss uses puids so a doc that is a gold
          of query q is scored as a POSITIVE for q even when it sits in another
          query's positive slot in the same batch (true multi-positive, no false neg).
"""
import json, os, collections, random

FT = "train-data"                            # pairs.jsonl + train_ecfr.jsonl + train_synth.jsonl
EVAL_A = "eval-data/questions.jsonl"         # Set A questions (with train/eval split)
EVAL_B = "eval-data/questions_open.jsonl"    # Set B questions (with train/eval split)
PMAX, NNEG = 3, 5
rng = random.Random(0)

# 1) group pairs.jsonl by query -> ordered unique positives keyed by pos_uuid
groups = collections.OrderedDict()
uuid2id = {}
for l in open(f"{FT}/pairs.jsonl"):
    r = json.loads(l)
    g = groups.setdefault(r["query"], {"pos": collections.OrderedDict(), "source": r["source"]})
    if r["pos_uuid"] not in g["pos"]:
        g["pos"][r["pos_uuid"]] = r["positive"]
    if r["pos_uuid"] not in uuid2id:
        uuid2id[r["pos_uuid"]] = len(uuid2id)

# 2) anchor -> mined hard negatives (union across both files, dedup)
negmap = collections.defaultdict(list)
for fn in ("train_ecfr.jsonl", "train_synth.jsonl"):
    path = f"{FT}/{fn}"
    if not os.path.exists(path):
        continue
    for l in open(path):
        r = json.loads(l)
        seen = set(negmap[r["anchor"]])
        for k in range(1, NNEG + 1):
            t = r.get(f"negative_{k}")
            if t and t not in seen:
                negmap[r["anchor"]].append(t); seen.add(t)

# 3) holdout disjointness — eval question texts (Set A eval split + Set B eval split)
eval_texts = {json.loads(l)["question"] for l in open(EVAL_A)
              if json.loads(l).get("split") == "eval"}
if os.path.exists(EVAL_B):
    eval_texts |= {json.loads(l)["question"] for l in open(EVAL_B) if json.loads(l).get("split") == "eval"}

all_neg_pool = []
for q in groups:
    all_neg_pool.extend(negmap.get(q, [])[:NNEG])

rows, leak, noneg, npos_hist = [], 0, 0, collections.Counter()
for q, g in groups.items():
    if q in eval_texts:
        leak += 1; continue
    pos_uuids = list(g["pos"].keys())
    pos_texts = list(g["pos"].values())
    pos = pos_texts[:PMAX]
    puids = [uuid2id[u] for u in pos_uuids[:PMAX]]
    pos_set = set(pos_texts)
    n_pos = len(pos)
    npos_hist[n_pos] += 1
    pos_cols = (pos + [pos[0]] * PMAX)[:PMAX]                     # pad with copy of pos_1
    puid_cols = (puids + [-1] * PMAX)[:PMAX]                      # pad puid = -1
    negs = [n for n in negmap.get(q, []) if n not in pos_set][:NNEG]
    if not negs:
        noneg += 1
    while len(negs) < NNEG:
        cand = rng.choice(all_neg_pool) if all_neg_pool else pos[0]
        if cand not in pos_set and cand not in negs:
            negs.append(cand)
    row = {"anchor": q}
    for i in range(PMAX):
        row[f"positive_{i+1}"] = pos_cols[i]
    for i in range(NNEG):
        row[f"negative_{i+1}"] = negs[i]
    row["label"] = [n_pos] + puid_cols                            # [n_pos, puid1, puid2, puid3]
    row["src"] = "ecfr" if g["source"].startswith("ecfr") else "open"  # picks the eval-matching instruction
    rows.append(row)

rng.shuffle(rows)
out = f"{FT}/train_v3.jsonl"
with open(out, "w") as fo:
    for r in rows:
        fo.write(json.dumps(r) + "\n")

print(f"wrote {len(rows)} rows -> {out}  | unique gold uuids: {len(uuid2id)}")
print(f"n_pos histogram: {dict(sorted(npos_hist.items()))}")
print(f"holdout leaks dropped: {leak}  | queries with no mined neg: {noneg}")
print(f"multi-positive queries (n_pos>=2): {sum(v for k,v in npos_hist.items() if k>=2)}")
