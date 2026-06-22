#!/usr/bin/env python3
"""Build a grounded question->governing-statute retrieval eval from the Nemotron
eCFR-QA + eCFR subsets.

Ground truth is *derived from the data*, not guessed: each eCFR-QA item cites the
CFR section its answer rests on (e.g. "1 CFR § 2.2(b)"); we map that citation to
the eCFR document that contains the section, and use it as the gold target. Every
citation/section/part token is then STRIPPED from the question, so retrieval must
be semantic rather than string-matching a section number.

Outputs
  questions.jsonl   {qid, question, gold_uuid, gold_title, gold_section, split}
  ecfr_corpus.jsonl {doc_uuid, title, sections, text}   (the retrieval corpus)

`split` partitions the reservoir into disjoint "eval" / "train" question sets so
that training pairs mined from "train" never leak into the "eval" measurement.
We deliberately over-generate (a reservoir) and use only a slice for eval.
"""
import argparse, glob, json, os, random, re
import pyarrow.parquet as pq

CIT   = re.compile(r'(\d+)\s*CFR\s*§\s*(\d+\.\d+)', re.I)
SEC   = re.compile(r'§\s*(\d+\.\d+)')
TITLE = re.compile(r'Title\s+(\d+)', re.I)
STRIP = re.compile(r'\s*(?:,)?\s*(?:under|according to|pursuant to|per|in|as (?:set forth|provided|described|stated) in|specified in|defined in|outlined in)?\s*\d+\s*CFR\s*§\s*\d+(?:\.\d+)?(?:\([a-zA-Z0-9]+\))*', re.I)


def build_corpus(data):
    idx, docs = {}, {}
    for f in sorted(glob.glob(os.path.join(data, "Nemotron-Pretraining-Legal-eCFR", "*.parquet"))):
        t = pq.read_table(f, columns=["text", "uuid"])
        for text, uuid in zip(t.column("text").to_pylist(), t.column("uuid").to_pylist()):
            text = str(text)
            mt = TITLE.search(text); title = int(mt.group(1)) if mt else None
            secs = sorted(set(SEC.findall(text)))
            docs[uuid] = {"title": title, "text": text, "sections": secs}
            for s in secs:
                idx.setdefault((title, s), []).append(uuid)
    return idx, docs


def clean_q(q):
    q = STRIP.sub('', q)
    q = re.sub(r'\s*(?:Subpart\s+[A-Z]\s+of\s+)?\d+\s*CFR\s*Part\s*\d+', '', q, flags=re.I)
    q = re.sub(r'\s*§+\s*\d+\.\d+(?:\([a-zA-Z0-9]+\))*', '', q)
    q = re.sub(r'\s*\bSubpart\s+[A-Z]\b', '', q)
    q = re.sub(r'\s*\b(?:under|of|in|per|pursuant to|according to)\s*([?.])', r'\1', q, flags=re.I)
    q = re.sub(r'\s{2,}', ' ', q)
    q = re.sub(r'\s+([?.,;:])', r'\1', q).strip(" ,;:")
    return q.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/datasets/Nemotron-Pretraining-Legal-v1"))
    ap.add_argument("--out", default="eval/questions.jsonl")
    ap.add_argument("--corpus-out", default="eval/ecfr_corpus.jsonl")
    ap.add_argument("--n", type=int, default=5000, help="reservoir size to emit")
    ap.add_argument("--eval-n", type=int, default=1000, help="how many of n to mark split=eval (rest=train)")
    ap.add_argument("--scan", type=int, default=800000, help="eCFR-QA rows to scan for candidates")
    ap.add_argument("--max-per-doc", type=int, default=2, help="cap questions per gold doc for diversity")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for p in (a.out, a.corpus_out):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    idx, docs = build_corpus(a.data)
    print(f"eCFR corpus: {len(docs)} docs, {len(idx)} (title,section) keys", flush=True)

    cands, seen = [], 0
    nocite = nomatch = multi = short = leak = 0
    percnt, stop = {}, False
    for f in sorted(glob.glob(os.path.join(a.data, "Nemotron-Pretraining-Legal-eCFR-QA", "*.parquet"))):
        if stop: break
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text", "uuid"]):
            for text, uuid in zip(b.column("text").to_pylist(), b.column("uuid").to_pylist()):
                seen += 1
                if seen > a.scan: stop = True; break
                text = str(text)
                qpart = re.split(r'\bAnswer:', text, maxsplit=1)[0].strip()
                m = CIT.search(qpart) or CIT.search(text)
                if not m: nocite += 1; continue
                title, section = int(m.group(1)), m.group(2)
                hits = idx.get((title, section))
                if not hits: nomatch += 1; continue
                if len(hits) > 1: multi += 1; continue
                gold = hits[0]
                if percnt.get(gold, 0) >= a.max_per_doc: continue
                qc = clean_q(qpart)
                if len(qc) < 25: short += 1; continue
                if '§' in qc or re.search(r'\bCFR\b', qc, re.I): leak += 1; continue
                percnt[gold] = percnt.get(gold, 0) + 1
                cands.append({"qid": f"ecfrqa-{uuid[:8]}", "question": qc, "gold_uuid": gold,
                              "gold_title": title, "gold_section": section, "src_uuid": uuid})
            if stop: break
    print(f"scanned {seen} | no-cite {nocite} | no-match {nomatch} | ambiguous {multi} | "
          f"too-short {short} | leak {leak} | usable {len(cands)}", flush=True)
    if len(cands) < a.n:
        print(f"WARNING: only {len(cands)} usable < requested {a.n}; raise --scan", flush=True)

    random.Random(a.seed).shuffle(cands)
    sel = cands[:a.n]
    for i, c in enumerate(sel):
        c["split"] = "eval" if i < a.eval_n else "train"
    with open(a.out, "w") as fo:
        for c in sel: fo.write(json.dumps(c) + "\n")
    with open(a.corpus_out, "w") as fo:
        for uuid, d in docs.items():
            fo.write(json.dumps({"doc_uuid": uuid, "title": d["title"],
                                 "sections": d["sections"], "text": d["text"][:6000]}) + "\n")
    ev = sum(1 for c in sel if c["split"] == "eval")
    print(f"wrote {len(sel)} questions ({ev} eval / {len(sel)-ev} train) -> {a.out}", flush=True)
    print(f"wrote {len(docs)} corpus docs -> {a.corpus_out}", flush=True)


if __name__ == "__main__":
    main()
