#!/usr/bin/env python3
"""Build (query, positive) training pairs for the legal embedding fine-tune:
  (1) eCFR-QA -> eCFR section  (citation-grounded, excluding Set-A eval qids)
  (2) Set B synthetic TRAIN split (question -> gold doc(s); multi-doc -> multi-positive)
Output: pairs.jsonl {query, positive, pos_uuid, pos_subset, source}. Hard negatives
are added in a separate step (mine_hard_negs.py)."""
import argparse, glob, json, os, re, random
from collections import Counter
import pyarrow.parquet as pq

CIT = re.compile(r'(\d+)\s*CFR\s*§\s*(\d+\.\d+)', re.I)
SEC = re.compile(r'§\s*(\d+\.\d+)'); TITLE = re.compile(r'Title\s+(\d+)', re.I)
STRIP = re.compile(r'\s*(?:,)?\s*(?:under|according to|pursuant to|per|in|as (?:set forth|provided|described|stated) in|specified in|defined in|outlined in)?\s*\d+\s*CFR\s*§\s*\d+(?:\.\d+)?(?:\([a-zA-Z0-9]+\))*', re.I)


def clean_q(q):
    q = STRIP.sub('', q)
    q = re.sub(r'\s*(?:Subpart\s+[A-Z]\s+of\s+)?\d+\s*CFR\s*Part\s*\d+', '', q, flags=re.I)
    q = re.sub(r'\s*§+\s*\d+\.\d+(?:\([a-zA-Z0-9]+\))*', '', q)
    q = re.sub(r'\s*\bSubpart\s+[A-Z]\b', '', q)
    q = re.sub(r'\s*\b(?:under|of|in|per|pursuant to|according to)\s*([?.])', r'\1', q, flags=re.I)
    q = re.sub(r'\s{2,}', ' ', q); q = re.sub(r'\s+([?.,;:])', r'\1', q).strip(" ,;:")
    return q.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/datasets/Nemotron-Pretraining-Legal-v1"))
    ap.add_argument("--eval-dir", default="eval")
    ap.add_argument("--open-dir", default="eval-open")
    ap.add_argument("--out", default="train-data/pairs.jsonl")
    ap.add_argument("--ecfr-target", type=int, default=40000)
    ap.add_argument("--scan", type=int, default=400000)
    ap.add_argument("--max-per-doc", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    excl = {json.loads(l)["qid"] for l in open(os.path.join(a.eval_dir, "questions.jsonl"))
            if json.loads(l)["split"] == "eval"}
    # also exclude by exact query TEXT (a held-out question can recur under a different uuid)
    eval_texts = {json.loads(l)["question"] for l in open(os.path.join(a.eval_dir, "questions.jsonl"))
                  if json.loads(l)["split"] == "eval"}
    _op = os.path.join(a.open_dir, "questions_open.jsonl")
    if os.path.exists(_op):
        eval_texts |= {json.loads(l)["question"] for l in open(_op) if json.loads(l).get("split") == "eval"}
    corp = [json.loads(l) for l in open(os.path.join(a.eval_dir, "ecfr_corpus.jsonl"))]
    ecorp = {d["doc_uuid"]: d["text"] for d in corp}
    idx = {}
    for d in corp:
        for s in d.get("sections", []):
            idx.setdefault((d["title"], s), []).append(d["doc_uuid"])

    pairs = []; percnt = {}; seen = ne = 0; stop = False
    for f in sorted(glob.glob(os.path.join(a.data, "Nemotron-Pretraining-Legal-eCFR-QA", "*.parquet"))):
        if stop: break
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text", "uuid"]):
            for text, uuid in zip(b.column("text").to_pylist(), b.column("uuid").to_pylist()):
                seen += 1
                if seen > a.scan or ne >= a.ecfr_target: stop = True; break
                if f"ecfrqa-{uuid[:8]}" in excl: continue
                text = str(text); qp = re.split(r'\bAnswer:', text, maxsplit=1)[0].strip()
                m = CIT.search(qp) or CIT.search(text)
                if not m: continue
                hits = idx.get((int(m.group(1)), m.group(2)))
                if not hits or len(hits) > 1: continue
                gold = hits[0]
                if percnt.get(gold, 0) >= a.max_per_doc: continue
                qc = clean_q(qp)
                if len(qc) < 25 or '§' in qc or re.search(r'\bCFR\b', qc, re.I): continue
                if qc in eval_texts: continue
                percnt[gold] = percnt.get(gold, 0) + 1; ne += 1
                pairs.append({"query": qc, "positive": ecorp[gold][:a.max_chars],
                              "pos_uuid": gold, "pos_subset": "eCFR", "source": "ecfr-qa"})
            if stop: break
    print(f"eCFR-QA->eCFR pairs: {ne}", flush=True)

    u2t, u2s = dict(ecorp), {u: "eCFR" for u in ecorp}
    for fn in ["single.jsonl", "clusters_sibling.jsonl", "clusters_semantic.jsonl"]:
        p = os.path.join(a.open_dir, fn)
        if not os.path.exists(p): continue
        for r in (json.loads(l) for l in open(p)):
            for m in (r["members"] if "members" in r else [r]):
                u2t[m["uuid"]] = m["text"]; u2s[m["uuid"]] = m["subset"]
    nb = 0
    for r in (json.loads(l) for l in open(os.path.join(a.open_dir, "questions_open.jsonl"))):
        if r.get("split") != "train" or r["question"] in eval_texts: continue
        for g in r["gold_uuids"]:
            if g in u2t:
                pairs.append({"query": r["question"], "positive": u2t[g][:a.max_chars], "pos_uuid": g,
                              "pos_subset": u2s.get(g, "?"), "source": "synthetic-" + r["question_kind"]}); nb += 1
    print(f"Set B train pairs (multi-positive expanded): {nb}", flush=True)

    random.Random(a.seed).shuffle(pairs)
    with open(a.out, "w") as fo:
        for p in pairs: fo.write(json.dumps(p) + "\n")
    print(f"wrote {len(pairs)} pairs -> {a.out}")
    print("by source:", dict(Counter(p["source"] for p in pairs)))
    print("by pos_subset:", dict(Counter(p["pos_subset"] for p in pairs)))


if __name__ == "__main__":
    main()
