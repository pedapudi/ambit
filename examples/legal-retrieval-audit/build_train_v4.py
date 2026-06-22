#!/usr/bin/env python3
"""v4 — build a BALANCED multi-positive training set (~1/3 multi-document).

The v3 post-mortem found the multi-hop gap is data composition: only ~3.7% of v3's
training queries were multi-positive. v4 fixes the ratio. Sources:
  train-data/pairs.jsonl        existing eCFR single-target + Set B (single + multi) pairs
  train-data/train_v4new.jsonl  NEW authored multi-doc questions (gold_texts + 5 hard negs)
Hard negs: train_ecfr.jsonl + train_synth.jsonl (existing) + train_v4new (new).
Caps the eCFR single-target queries so multi-positive == ~TARGET of the final set.
Holdout-disjoint from both eval splits (question text). Schema identical to train_v3:
  anchor, positive_1..3, negative_1..5, label=[n_pos, puid1..3], src.
"""
import json, os, collections, random

FT = "train-data"
EVAL_A = "eval-data/questions.jsonl"
EVAL_B = "eval-data/questions_open.jsonl"
PMAX, NNEG = 3, 5
TARGET_MULTI = 1.0 / 3.0          # single budget = (1/TARGET - 1) * multi  ==  2*multi
rng = random.Random(0)

# eval holdout question texts (Set A eval + Set B eval)
evq = {json.loads(l)["question"] for l in open(EVAL_A) if json.loads(l).get("split") == "eval"}
if os.path.exists(EVAL_B):
    evq |= {json.loads(l)["question"] for l in open(EVAL_B) if json.loads(l).get("split") == "eval"}

uuid2id = {}
def pid(u):
    if u not in uuid2id:
        uuid2id[u] = len(uuid2id)
    return uuid2id[u]

# 1) groups from pairs.jsonl  (query -> ordered unique positives keyed by pos_uuid)
groups = collections.OrderedDict()
for l in open(f"{FT}/pairs.jsonl"):
    r = json.loads(l)
    g = groups.setdefault(r["query"], {"pos": collections.OrderedDict(), "source": r["source"]})
    g["pos"].setdefault(r["pos_uuid"], r["positive"])

# 2) groups from train_v4new (the NEW multi-doc questions) + their per-question negs
v4groups, v4negs = collections.OrderedDict(), {}
for l in open(f"{FT}/train_v4new.jsonl"):
    r = json.loads(l)
    q = r["question"]
    g = v4groups.setdefault(q, {"pos": collections.OrderedDict(), "source": "open_v4"})
    for u, t in zip(r["gold_uuids"], r["gold_texts"]):
        g["pos"].setdefault(u, t)
    v4negs[q] = [r.get(f"negative_{k}") for k in range(1, NNEG + 1) if r.get(f"negative_{k}")]

# 3) negmap (anchor -> mined hard negs) from existing files, then v4 (per-question)
negmap = collections.defaultdict(list)
for fn in ("train_ecfr.jsonl", "train_synth.jsonl"):
    p = f"{FT}/{fn}"
    if not os.path.exists(p):
        continue
    for l in open(p):
        r = json.loads(l); seen = set(negmap[r["anchor"]])
        for k in range(1, NNEG + 1):
            t = r.get(f"negative_{k}")
            if t and t not in seen:
                negmap[r["anchor"]].append(t); seen.add(t)
for q, ns in v4negs.items():
    negmap[q] = ns

# 4) classify groups + holdout-exclude
def entry(q, g):
    return {"q": q, "pos_texts": list(g["pos"].values()), "pos_uuids": list(g["pos"].keys()),
            "n_pos": len(g["pos"]), "source": g["source"]}

ecfr_single, openb_single, multi, leak = [], [], [], 0
for q, g in list(groups.items()) + list(v4groups.items()):
    if q in evq:
        leak += 1; continue
    e = entry(q, g)
    if e["n_pos"] >= 2:
        multi.append(e)
    elif e["source"].startswith("ecfr"):
        ecfr_single.append(e)
    else:
        openb_single.append(e)

# 5) cap eCFR so multi == TARGET:  single_budget = (1/TARGET - 1) * multi
M = len(multi); s_b = len(openb_single)
single_budget = int(round((1.0 / TARGET_MULTI - 1.0) * M))
keep_ecfr = max(0, single_budget - s_b)
rng.shuffle(ecfr_single)
ecfr_kept = ecfr_single[:keep_ecfr]
final = multi + openb_single + ecfr_kept
rng.shuffle(final)

all_neg_pool = []
for e in final:
    all_neg_pool.extend(negmap.get(e["q"], [])[:NNEG])

rows, npos_hist, noneg = [], collections.Counter(), 0
for e in final:
    pos = e["pos_texts"][:PMAX]; puids = [pid(u) for u in e["pos_uuids"][:PMAX]]
    n_pos = len(pos); npos_hist[n_pos] += 1; pos_set = set(e["pos_texts"])
    pos_cols = (pos + [pos[0]] * PMAX)[:PMAX]; puid_cols = (puids + [-1] * PMAX)[:PMAX]
    negs = [n for n in negmap.get(e["q"], []) if n not in pos_set][:NNEG]
    if not negs:
        noneg += 1
    while len(negs) < NNEG:
        cand = rng.choice(all_neg_pool) if all_neg_pool else pos[0]
        if cand not in pos_set and cand not in negs:
            negs.append(cand)
    row = {"anchor": e["q"]}
    for i in range(PMAX):
        row[f"positive_{i+1}"] = pos_cols[i]
    for i in range(NNEG):
        row[f"negative_{i+1}"] = negs[i]
    row["label"] = [n_pos] + puid_cols
    row["src"] = "ecfr" if e["source"].startswith("ecfr") else "open"
    rows.append(row)

rng.shuffle(rows)
out = f"{FT}/train_v4.jsonl"
with open(out, "w") as fo:
    for r in rows:
        fo.write(json.dumps(r) + "\n")

n = len(rows); nmulti = sum(v for k, v in npos_hist.items() if k >= 2)
v4_multi = sum(1 for e in multi if e["source"] == "open_v4")
print(f"wrote {n} rows -> {out}")
print(f"composition: multi(n_pos>=2)={nmulti} ({100*nmulti/n:.1f}%) | single={n-nmulti}")
print(f"  multi = SetB-multi {M - v4_multi} + v4-new {v4_multi}")
print(f"  single = eCFR kept {len(ecfr_kept)}/{len(ecfr_single)} + SetB-single {s_b}")
print(f"n_pos histogram: {dict(sorted(npos_hist.items()))}")
print(f"unique gold uuids: {len(uuid2id)} | holdout leaks dropped: {leak} | rows w/ no mined neg: {noneg}")
