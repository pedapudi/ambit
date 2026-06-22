#!/usr/bin/env python3
"""Mine hard negatives for the eCFR training pairs: per query, retrieve the nearest
eCFR sections with the BASE model and take the top non-gold hits (the confusable
siblings — the within-family discrimination the model fails on). Outputs:
  train_ecfr.jsonl   {anchor, positive, negative_1..k}   (eCFR pairs + hard negs)
  train_synth.jsonl  {anchor, positive}                  (Set B pairs; in-batch negs)
anchor is the RAW question; the query instruction is applied uniformly at train time."""
import argparse, json, os, time
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor

INSTRUCT = ("Given a question about U.S. federal regulations, retrieve the Code of "
            "Federal Regulations section that governs the answer.")


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
    ap.add_argument("--ecfr-npy", default="eval/cache/ecfr_corpus.npy")
    ap.add_argument("--ecfr-meta", default="eval/cache/ecfr_corpus.meta.json")
    ap.add_argument("--ecfr-corpus", default="eval/ecfr_corpus.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--out-dir", default="train-data")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    model = requests.get(a.base_url + "/models", timeout=30).json()["data"][0]["id"]

    pairs = [json.loads(l) for l in open(a.pairs)]
    ecfr = [p for p in pairs if p["pos_subset"] == "eCFR"]
    synth = [p for p in pairs if p["pos_subset"] != "eCFR"]
    print(f"eCFR pairs: {len(ecfr)} | synth pairs: {len(synth)}", flush=True)

    ids = json.load(open(a.ecfr_meta))["ids"]; D = np.load(a.ecfr_npy)   # normalized
    pos = {u: i for i, u in enumerate(ids)}
    text = {d["doc_uuid"]: d["text"] for d in (json.loads(l) for l in open(a.ecfr_corpus))}

    t0 = time.time()
    Q = embed([f"Instruct: {INSTRUCT}\nQuery: {p['query']}" for p in ecfr], a.base_url, model)
    print(f"embedded {len(Q)} queries in {time.time()-t0:.0f}s", flush=True)

    fo = open(os.path.join(a.out_dir, "train_ecfr.jsonl"), "w"); kept = 0
    for s in range(0, len(ecfr), 20000):
        sims = Q[s:s + 20000] @ D.T
        for j, p in enumerate(ecfr[s:s + 20000]):
            gi = pos.get(p["pos_uuid"])
            top = np.argpartition(-sims[j], a.k + 2)[:a.k + 2]
            top = top[np.argsort(-sims[j][top])]
            negs = [text[ids[di]][:6000] for di in top if di != gi][:a.k]
            row = {"anchor": p["query"], "positive": p["positive"]}
            for n, tn in enumerate(negs): row[f"negative_{n+1}"] = tn
            fo.write(json.dumps(row) + "\n"); kept += 1
    fo.close()
    with open(os.path.join(a.out_dir, "train_synth.jsonl"), "w") as fo:
        for p in synth: fo.write(json.dumps({"anchor": p["query"], "positive": p["positive"]}) + "\n")
    print(f"wrote train_ecfr.jsonl ({kept}, k={a.k}) + train_synth.jsonl ({len(synth)}) -> {a.out_dir}")


if __name__ == "__main__":
    main()
