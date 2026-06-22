#!/usr/bin/env python3
"""Open-corpus retrieval eval for the synthetic question set (questions_open.jsonl).

Corpus = the 1M map embeddings (out-1024-1M: CA-Regs, Case-Law-Summary, CaseHOLD,
eCFR-QA) + the 35k eCFR section embeddings — ~1.035M docs, all already computed.
Questions are embedded (with a query instruction) and CACHED. Reports, split by
question_kind:
  single-doc : Recall@k, MRR@10, median rank
  multi-doc  : set-Recall@k (fraction of gold docs in top-k) and all-gold@k
"""
import argparse, glob, json, os, time
import numpy as np, requests, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

INSTRUCT = "Given a question about U.S. law and regulations, retrieve the document(s) that answer it."


def get_model(base):
    return requests.get(base + "/models", timeout=30).json()["data"][0]["id"]


def embed_all(texts, base, model, batch=32, workers=8, max_chars=8000):
    sess = requests.Session()
    texts = [(t[:max_chars] if t and t.strip() else " ") for t in texts]

    def one(tb):
        for attempt in range(6):
            r = sess.post(base + "/embeddings", json={"model": model, "input": tb}, timeout=300)
            if r.status_code == 200:
                return [d["embedding"] for d in sorted(r.json()["data"], key=lambda x: x["index"])]
            if r.status_code in (429, 500, 502, 503): time.sleep(1.5 * (attempt + 1)); continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("retries exhausted")
    bs = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, bs))
    v = np.asarray([x for vb in out for x in vb], dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


def cached_embed(tag, texts, ids, cache_dir, base, model):
    npy = os.path.join(cache_dir, tag + ".npy"); meta = os.path.join(cache_dir, tag + ".meta.json")
    if os.path.exists(npy) and os.path.exists(meta) and json.load(open(meta)).get("ids") == ids \
       and json.load(open(meta)).get("model") == model:
        print(f"  cache hit {tag}", flush=True); return np.load(npy)
    t0 = time.time(); v = embed_all(texts, base, model); os.makedirs(cache_dir, exist_ok=True)
    np.save(npy, v); json.dump({"model": model, "ids": ids}, open(meta, "w"))
    print(f"  embedded+cached {tag}: {v.shape} in {time.time()-t0:.0f}s", flush=True); return v


def build_corpus(map_dir, ecfr_npy, ecfr_meta):
    uuids, mats = [], []
    for f in sorted(glob.glob(map_dir + "/*.parquet")):
        t = pq.read_table(f, columns=["uuid", "embedding"])
        uuids += t.column("uuid").to_pylist()
        mats.append(t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(t), -1))
    ids = json.load(open(ecfr_meta))["ids"]; uuids += ids; mats.append(np.load(ecfr_npy))
    D = np.vstack(mats); D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return D, {u: i for i, u in enumerate(uuids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval-open")
    ap.add_argument("--map-dir", default="emb-1024-1M")
    ap.add_argument("--ecfr-npy", default="eval/cache/ecfr_corpus.npy")
    ap.add_argument("--ecfr-meta", default="eval/cache/ecfr_corpus.meta.json")
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    cache_dir = os.path.join(a.eval_dir, "cache")
    model = get_model(a.base_url)

    qs = [json.loads(l) for l in open(os.path.join(a.eval_dir, "questions_open.jsonl"))]
    print(f"model: {model} | questions: {len(qs)}", flush=True)
    D, pos = build_corpus(a.map_dir, a.ecfr_npy, a.ecfr_meta)
    print(f"corpus: {D.shape}", flush=True)

    Q = cached_embed("queries_open.instr", [f"Instruct: {INSTRUCT}\nQuery: {q['question']}" for q in qs],
                     [q["qid"] for q in qs], cache_dir, a.base_url, model)

    sel = [i for i, q in enumerate(qs) if a.split == "all" or q.get("split") == a.split]
    # gold instances among selected queries whose gold docs are all in the corpus
    inst_q, inst_thr, inst_key = [], [], []   # key = (query position in sel, kind)
    skipped = 0
    Qsel = Q[sel]
    qmap = {gi: k for k, gi in enumerate(sel)}
    for k, i in enumerate(sel):
        golds = qs[i]["gold_uuids"]
        if any(g not in pos for g in golds): skipped += 1; continue
        for g in golds:
            inst_q.append(k); inst_thr.append(float(Qsel[k] @ D[pos[g]]))
            inst_key.append((k, qs[i]["question_kind"]))
    inst_q = np.array(inst_q); inst_thr = np.array(inst_thr, dtype=np.float32)
    cnt = np.zeros(len(inst_q), dtype=np.int64)
    for c in range(0, D.shape[0], 20000):
        P = Qsel @ D[c:c + 20000].T          # (n_sel, c)
        cnt += (P[inst_q] > inst_thr[:, None]).sum(1)
    ranks = cnt + 1

    # regroup ranks per query
    per_q = {}
    for (k, kind), r in zip(inst_key, ranks):
        per_q.setdefault(k, [kind, []])[1].append(int(r))

    single = [v[1][0] for v in per_q.values() if v[0] == "single-doc"]
    multi = [v[1] for v in per_q.values() if v[0] == "multi-doc"]
    print(f"\nevaluated split='{a.split}': {len(per_q)} queries ({len(single)} single, {len(multi)} multi); "
          f"{skipped} skipped (gold not in corpus)\n")

    def at(ks, ranks_list): return {f"R@{k}": round(np.mean([r <= k for r in ranks_list]), 3) for k in ks}
    if single:
        s = np.array(single)
        print(f"[single-doc] {at((1,5,10,100), s)} | MRR@10 {round(float(np.mean([1/r if r<=10 else 0 for r in s])),3)} "
              f"| median {int(np.median(s))}")
    if multi:
        for k in (5, 10, 20, 100):
            setrec = np.mean([np.mean([r <= k for r in rs]) for rs in multi])
            allk = np.mean([all(r <= k for r in rs) for rs in multi])
            print(f"[multi-doc] set-Recall@{k} {setrec:.3f} | all-gold@{k} {allk:.3f}")


if __name__ == "__main__":
    main()
