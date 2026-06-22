#!/usr/bin/env python3
"""Sample source documents (from subdomains whose embeddings already exist in the
1M map / eCFR corpus) for the OPEN-CORPUS synthetic eval. Emits candidates.jsonl
{uuid, subset, text} that LLM subagents read to author one grounded question
per doc (gold = that uuid). Only embedded uuids are sampled so gold is retrievable."""
import argparse, glob, json, os, random
import pyarrow.parquet as pq

DATA = os.path.expanduser("~/datasets/Nemotron-Pretraining-Legal-v1")
MAP  = "emb-1024-1M"

def sample_from_dataset(subset, n, rng, max_chars):
    """reservoir-sample (uuid,text) from a dataset subdomain (all docs embedded)."""
    files = sorted(glob.glob(os.path.join(DATA, subset, "*.parquet")))
    res, seen = [], 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text", "uuid"]):
            for t, u in zip(b.column("text").to_pylist(), b.column("uuid").to_pylist()):
                seen += 1; item = (u, str(t)[:max_chars])
                if len(res) < n: res.append(item)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < n: res[j] = item
    return res

def embedded_uuids(prefix):
    """uuids actually embedded in the 1M map for a given chunk prefix (e.g. casehold)."""
    us = set()
    for f in sorted(glob.glob(os.path.join(MAP, prefix + "-*.parquet"))):
        us.update(pq.read_table(f, columns=["uuid"]).column("uuid").to_pylist())
    return us

def text_for(subset, uuids, max_chars):
    want = set(uuids); out = {}
    for f in sorted(glob.glob(os.path.join(DATA, subset, "*.parquet"))):
        b = pq.read_table(f, columns=["text", "uuid"])
        for t, u in zip(b.column("text").to_pylist(), b.column("uuid").to_pylist()):
            if u in want: out[u] = str(t)[:max_chars]
        if len(out) >= len(want): break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval-open/candidates.jsonl")
    ap.add_argument("--per-subset", type=int, default=1400)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(a.seed)

    cands = []
    # fully-embedded subdomains: sample straight from the dataset
    for sub in ["Nemotron-Pretraining-Legal-California-Code-Of-Regulations",
                "Nemotron-Pretraining-Legal-Case-Law-Summary",
                "Nemotron-Pretraining-Legal-eCFR"]:
        for u, t in sample_from_dataset(sub, a.per_subset, rng, a.max_chars):
            cands.append({"uuid": u, "subset": sub.replace("Nemotron-Pretraining-Legal-", ""), "text": t})
        print(f"sampled {sub.split('-')[-1]}", flush=True)
    # CaseHOLD: only a subset embedded -> sample from embedded uuids, then fetch text
    ch = list(embedded_uuids("casehold")); rng.shuffle(ch); ch = ch[:a.per_subset]
    txt = text_for("Nemotron-Pretraining-Legal-CaseHOLD", ch, a.max_chars)
    for u in ch:
        if u in txt: cands.append({"uuid": u, "subset": "CaseHOLD", "text": txt[u]})
    print(f"sampled CaseHOLD ({len(txt)})", flush=True)

    rng.shuffle(cands)
    with open(a.out, "w") as fo:
        for c in cands: fo.write(json.dumps(c) + "\n")
    print(f"wrote {len(cands)} candidates -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
