#!/usr/bin/env python3
"""Merge authored question files into a single, fully-labeled open-corpus eval set,
attaching the structural labels that make the set self-describing, validating
against citation leakage, and assigning a deterministic train/eval split.

Inputs
  --authored-dir   dir of *.out.jsonl written by the authoring agents
                   single: {uuid, question, reasoning_type, note}
                   multi:  {cluster_id, gold_uuids, question, reasoning_type, note, cluster_type}
  --single / --sibling / --semantic   the input pools (for uuid->subset lookup)

Output  questions_open.jsonl, one record per question:
  qid, question, gold_uuids[], n_gold, question_kind, grounding,
  source_subsets[], reasoning_type, note, split
and prints the full label distribution so the set is understood at a glance.
"""
import argparse, glob, json, os, random, re

# Citation/locator leakage only — NOT bare decimals (those are legitimate content,
# e.g. numeric thresholds). Drop a question only if it names a legal locator.
LEAK = re.compile(r'§|\bCFR\b|\bU\.?S\.?C\.?\b|\bPart\s+\d+|\bsection\s+\d+', re.I)


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authored-dir", default="eval-open/authored")
    ap.add_argument("--single", default="eval-open/single.jsonl")
    ap.add_argument("--sibling", default="eval-open/clusters_sibling.jsonl")
    ap.add_argument("--semantic", default="eval-open/clusters_semantic.jsonl")
    ap.add_argument("--out", default="eval-open/questions_open.jsonl")
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    uuid_subset = {r["uuid"]: r["subset"] for r in load_jsonl(a.single)}
    for cf in (a.sibling, a.semantic):
        for c in load_jsonl(cf):
            for m in c["members"]:
                uuid_subset[m["uuid"]] = m["subset"]

    out, dropped = [], 0
    for f in sorted(glob.glob(os.path.join(a.authored_dir, "*.out.jsonl"))):
        is_multi = os.path.basename(f).startswith("multi")
        for r in load_jsonl(f):
            q = r.get("question", "").strip()
            if not q or LEAK.search(q):
                dropped += 1; continue
            if is_multi:
                gold = r.get("gold_uuids", [])
                grounding = {"sibling": "sibling-cluster", "semantic": "semantic-cluster"}.get(r.get("cluster_type"), "cluster")
                kind = "multi-doc"
            else:
                gold = [r["uuid"]]
                grounding = "synthetic-single"
                kind = "single-doc"
            if not gold or any(g not in uuid_subset for g in gold):
                dropped += 1; continue
            subs = sorted({uuid_subset[g] for g in gold})
            out.append({"question": q, "gold_uuids": gold, "n_gold": len(gold),
                        "question_kind": kind, "grounding": grounding, "source_subsets": subs,
                        "reasoning_type": r.get("reasoning_type", "unknown"), "note": r.get("note", "")})

    rng = random.Random(a.seed); rng.shuffle(out)
    # stratified split by (kind, grounding)
    strata = {}
    for r in out: strata.setdefault((r["question_kind"], r["grounding"]), []).append(r)
    final = []
    for k, rows in strata.items():
        ne = int(round(len(rows) * a.eval_frac))
        for i, r in enumerate(rows): r["split"] = "eval" if i < ne else "train"
        final += rows
    rng.shuffle(final)
    for i, r in enumerate(final):
        r["qid"] = f"open-{'m' if r['question_kind']=='multi-doc' else 's'}-{i:05d}"
    with open(a.out, "w") as fo:
        for r in final: fo.write(json.dumps(r) + "\n")

    # ---- label distribution (the "well-understood" summary) ----
    from collections import Counter
    def dist(key): return dict(Counter(r[key] for r in final).most_common())
    print(f"wrote {len(final)} questions ({dropped} dropped: empty/leak/bad-gold) -> {a.out}")
    print("kind:      ", dist("question_kind"))
    print("grounding: ", dist("grounding"))
    print("reasoning: ", dist("reasoning_type"))
    print("split:     ", dist("split"))
    print("n_gold:    ", dict(Counter(r["n_gold"] for r in final).most_common()))
    print("source_subsets (multi):", dict(Counter("+".join(r["source_subsets"]) for r in final if r["question_kind"]=="multi-doc").most_common(8)))


if __name__ == "__main__":
    main()
