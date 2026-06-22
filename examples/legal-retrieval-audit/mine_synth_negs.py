#!/usr/bin/env python3
"""v2 — mine hard negatives for the Set B (synthetic) training pairs so they match
the eCFR schema {anchor, positive, negative_1..k}. This is the change that turns the
two training sources into ONE uniform-shape dataset, which multi-GPU DDP requires
(mixing a 7-column dataset with a 2-column one desyncs the ranks → NCCL hang; see
FINETUNE.md, "Multi-GPU lessons" #2).

For each synthetic question, retrieve the nearest docs from a TEXT-available pool
(the eCFR sections + the Set B candidate docs) with the BASE model and take the top
non-gold hits, EXCLUDING the question's FULL gold set — so a sibling gold of a
multi-hop question is never mined as that question's own negative.

Overwrites train_synth.jsonl (the 2-column v1 file emitted by mine_hard_negs.py) with
the 7-column v2 version. anchor is the RAW question; the query instruction is applied
uniformly at train time."""
import argparse, glob, json, os, time
import numpy as np, requests, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

INSTRUCT = "Given a question about U.S. law and regulations, retrieve the document(s) that answer it."


def embed(texts, base, model, batch=32, workers=8, mc=8000):
    sess = requests.Session(); texts = [t[:mc] if t.strip() else " " for t in texts]
    def one(tb):
        for _ in range(6):
            r = sess.post(base + "/embeddings", json={"model": model, "input": tb}, timeout=300)
            if r.status_code == 200:
                return [d["embedding"] for d in sorted(r.json()["data"], key=lambda x: x["index"])]
            time.sleep(1.5)
        raise RuntimeError(r.text[:200])
    bs = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, bs))
    v = np.asarray([x for vb in out for x in vb], dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9); return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="train-data/pairs.jsonl")
    ap.add_argument("--questions-open", default="eval-data/questions_open.jsonl",
                    help="Set B questions, for the FULL gold set per question (gold-exclusion)")
    ap.add_argument("--ecfr-npy", default="eval/cache/ecfr_corpus.npy")
    ap.add_argument("--ecfr-meta", default="eval/cache/ecfr_corpus.meta.json")
    ap.add_argument("--ecfr-corpus", default="eval/ecfr_corpus.jsonl")
    ap.add_argument("--candidates", default="eval-data/candidates.jsonl",
                    help="Set B candidate docs (uuid, text) from build_open_candidates.py")
    ap.add_argument("--map-dir", default="map-1M",
                    help="parquet dir of BASE-model map embeddings (uuid, embedding) for the candidates")
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--out-dir", default="train-data")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    model = requests.get(a.base_url + "/models", timeout=30).json()["data"][0]["id"]

    # synth training pairs = everything whose positive is not an eCFR section
    synth = [p for p in (json.loads(l) for l in open(a.pairs)) if p["pos_subset"] != "eCFR"]

    # pool part 1: eCFR sections (text + cached base embeddings, already normalized)
    ids = json.load(open(a.ecfr_meta))["ids"]; V = np.load(a.ecfr_npy)
    etext = {d["doc_uuid"]: d["text"] for d in (json.loads(l) for l in open(a.ecfr_corpus))}
    pool_uuid = list(ids); pool_text = [etext[u][:6000] for u in ids]; mats = [V]

    # pool part 2: Set B candidate docs — text from candidates.jsonl, base embedding from the map
    ctext = {c["uuid"]: c["text"] for c in (json.loads(l) for l in open(a.candidates))}
    want = set(ctext) - set(ids); cmap = {}
    for f in sorted(glob.glob(os.path.join(a.map_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["uuid", "embedding"]); us = t.column("uuid").to_pylist()
        M = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)
        for u, vv in zip(us, M):
            if u in want and u not in cmap: cmap[u] = vv
        if want.issubset(cmap.keys()): break
    cu = [u for u in ctext if u in cmap]
    cm = np.stack([cmap[u] for u in cu]); cm /= (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-9)
    pool_uuid += cu; pool_text += [ctext[u][:6000] for u in cu]; mats.append(cm)
    D = np.vstack(mats)
    print(f"hard-neg pool: {D.shape} (eCFR {len(ids)} + candidates {len(cu)})", flush=True)

    # full gold set per question text, so sibling golds of a multi-hop question are excluded
    q2gold = {r["question"]: set(r["gold_uuids"]) for r in (json.loads(l) for l in open(a.questions_open))}
    Q = embed([f"Instruct: {INSTRUCT}\nQuery: {p['query']}" for p in synth], a.base_url, model)
    print(f"embedded {len(Q)} synth queries", flush=True)

    with open(os.path.join(a.out_dir, "train_synth.jsonl"), "w") as fo:
        for s in range(0, len(synth), 5000):
            sims = Q[s:s + 5000] @ D.T
            for j, p in enumerate(synth[s:s + 5000]):
                golds = q2gold.get(p["query"], set())
                top = np.argpartition(-sims[j], 20)[:20]; top = top[np.argsort(-sims[j][top])]
                negs = [pool_text[i] for i in top if pool_uuid[i] not in golds][:a.k]
                row = {"anchor": p["query"], "positive": p["positive"]}
                for n, tn in enumerate(negs): row[f"negative_{n+1}"] = tn
                fo.write(json.dumps(row) + "\n")
    print(f"wrote train_synth.jsonl ({len(synth)} rows, k={a.k}, uniform 7-col schema) -> {a.out_dir}")


if __name__ == "__main__":
    main()
