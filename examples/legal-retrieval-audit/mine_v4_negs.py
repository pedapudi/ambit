#!/usr/bin/env python3
"""v4 — mine 5 hard negatives for the NEW authored multi-doc training questions and
attach gold texts. Dedups authored questions vs the eval (question text AND gold-set,
belt-and-suspenders on the cluster-level exclusion in build_v4_inputs.py). Pool = eCFR
sections + the v4 candidate docs; excludes each question's FULL gold set so a sibling
gold is never mined as its own negative.

Input  : eval-open/authored_v4.jsonl  (subagent output, see AUTHORING_MULTI.md)
Output : train-data/train_v4new.jsonl  {question, gold_uuids, gold_texts, negative_1..5, source}
"""
import argparse, glob, json, os, time
import numpy as np, requests, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

OPEN_INSTRUCT = "Given a question about U.S. law and regulations, retrieve the document(s) that answer it."


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
    ap.add_argument("--authored", default="eval-open/authored_v4.jsonl")
    ap.add_argument("--clusters", default="eval-open/clusters_v4.jsonl", help="for gold-uuid -> text")
    ap.add_argument("--candidates", default="eval-open/candidates_v4.jsonl")
    ap.add_argument("--eval-a", default="eval-data/questions.jsonl")
    ap.add_argument("--eval-b", default="eval-data/questions_open.jsonl")
    ap.add_argument("--ecfr-npy", default="eval/cache/ecfr_corpus.npy")
    ap.add_argument("--ecfr-meta", default="eval/cache/ecfr_corpus.meta.json")
    ap.add_argument("--ecfr-corpus", default="eval/ecfr_corpus.jsonl")
    ap.add_argument("--map-dir", default="emb-1024-1M")
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--out", default="train-data/train_v4new.jsonl")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    model = requests.get(a.base_url + "/models", timeout=30).json()["data"][0]["id"]
    print("model:", model, flush=True)

    # gold uuid -> text (from the v4 clusters)
    uuid2text = {}
    for l in open(a.clusters):
        for m in json.loads(l)["members"]:
            uuid2text[m["uuid"]] = m["text"]

    # eval question texts + eval gold uuids (holdout disjointness)
    evq, evg = set(), set()
    for l in open(a.eval_a):
        r = json.loads(l)
        if r.get("split") == "eval":
            evq.add(r["question"]); evg.add(r["gold_uuid"])
    if os.path.exists(a.eval_b):
        for l in open(a.eval_b):
            r = json.loads(l)
            if r.get("split") == "eval":
                evq.add(r["question"]); evg.update(r["gold_uuids"])

    auth = [json.loads(l) for l in open(a.authored)]
    rows, drop_text, drop_gold, drop_notext, seen_q = [], 0, 0, 0, set()
    for x in auth:
        q = x["question"]
        if q in evq: drop_text += 1; continue
        if q in seen_q: continue
        if any(u in evg for u in x["gold_uuids"]): drop_gold += 1; continue
        gt = [uuid2text.get(u) for u in x["gold_uuids"]]
        if any(t is None for t in gt): drop_notext += 1; continue
        seen_q.add(q)
        rows.append({"question": q, "gold_uuids": x["gold_uuids"], "gold_texts": gt})
    print(f"authored {len(auth)} -> kept {len(rows)} | dropped: eval-text {drop_text}, eval-gold {drop_gold}, no-text {drop_notext}", flush=True)

    # pool: eCFR sections (cache) + v4 candidate docs (embeddings from the 1M map)
    ids = json.load(open(a.ecfr_meta))["ids"]; D0 = np.load(a.ecfr_npy)
    etext = {d["doc_uuid"]: d["text"] for d in (json.loads(l) for l in open(a.ecfr_corpus))}
    pool_uuid = list(ids); pool_text = [etext[u][:6000] for u in ids]; mats = [D0]
    ctext = {c["uuid"]: c["text"] for c in (json.loads(l) for l in open(a.candidates))}
    want = set(ctext) - set(ids); cmap = {}
    for f in sorted(glob.glob(a.map_dir + "/*.parquet")):
        t = pq.read_table(f, columns=["uuid", "embedding"]); us = t.column("uuid").to_pylist()
        M = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)
        for u, vv in zip(us, M):
            if u in want and u not in cmap: cmap[u] = vv
        if want.issubset(cmap.keys()): break
    cu = [u for u in ctext if u in cmap]
    cm = np.stack([cmap[u] for u in cu]); cm /= (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-9)
    pool_uuid += cu; pool_text += [ctext[u][:6000] for u in cu]; mats.append(cm)
    Dp = np.vstack(mats)
    print(f"hard-neg pool: {Dp.shape} (eCFR {len(ids)} + candidates {len(cu)})", flush=True)

    Q = embed([f"Instruct: {OPEN_INSTRUCT}\nQuery: {r['question']}" for r in rows], a.base_url, model)
    print(f"embedded {len(Q)} v4 queries", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fo:
        for s in range(0, len(rows), 5000):
            sims = Q[s:s + 5000] @ Dp.T
            for j, r in enumerate(rows[s:s + 5000]):
                golds = set(r["gold_uuids"])
                top = np.argpartition(-sims[j], 20)[:20]; top = top[np.argsort(-sims[j][top])]
                negs = [pool_text[i] for i in top if pool_uuid[i] not in golds][:a.k]
                row = {"question": r["question"], "gold_uuids": r["gold_uuids"],
                       "gold_texts": r["gold_texts"], "source": "open_v4"}
                for n, tn in enumerate(negs):
                    row[f"negative_{n+1}"] = tn
                fo.write(json.dumps(row) + "\n")
    print(f"wrote {a.out} ({len(rows)} rows, k={a.k})", flush=True)


if __name__ == "__main__":
    main()
