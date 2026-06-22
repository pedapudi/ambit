#!/usr/bin/env python3
"""Question->governing-statute retrieval eval for an OpenAI-compatible embedding
endpoint (e.g. vLLM / llama.cpp serving Qwen3-Embedding).

Embeds the eval questions (with an asymmetric query instruction) and the eCFR
corpus, then reports Recall@k / MRR / rank stats. Query AND corpus embeddings are
CACHED to disk (keyed by model + id list), so re-running an eval never re-embeds
unchanged text — only a new model or a changed question/corpus set triggers work.

Two corpus modes:
  closed  (default)  retrieve against the eCFR sections only — gold + siblings.
  open    (--distractor-dir DIR)  add other-domain docs as distractors (reusing
          already-computed map vectors), EXCLUDING --exclude-label to avoid the
          leakage of putting the questions' own source subset in the corpus.

Caches land in <eval-dir>/cache/.  Per-query ranks are written to baseline_ranks.json.
"""
import argparse, glob, json, os, time
import numpy as np, requests, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

INSTRUCT = ("Given a question about U.S. federal regulations, retrieve the Code of "
            "Federal Regulations section that governs the answer.")


def get_model(base):
    return requests.get(base + "/models", timeout=30).json()["data"][0]["id"]


def embed_all(texts, base, model, batch=32, workers=8, max_chars=8000):
    sess = requests.Session()
    texts = [(t[:max_chars] if t and t.strip() else " ") for t in texts]

    def one(tb):
        for attempt in range(6):
            r = sess.post(base + "/embeddings", json={"model": model, "input": tb}, timeout=300)
            if r.status_code == 200:
                d = sorted(r.json()["data"], key=lambda x: x["index"])
                return [x["embedding"] for x in d]
            if r.status_code in (429, 500, 502, 503): time.sleep(1.5 * (attempt + 1)); continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("retries exhausted")
    bs = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, bs))
    v = np.asarray([x for vb in out for x in vb], dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


def cached_embed(tag, texts, ids, cache_dir, base, model, **kw):
    npy = os.path.join(cache_dir, tag + ".npy")
    meta = os.path.join(cache_dir, tag + ".meta.json")
    if os.path.exists(npy) and os.path.exists(meta):
        m = json.load(open(meta))
        if m.get("model") == model and m.get("ids") == ids:
            v = np.load(npy)
            print(f"  cache hit {tag}: {v.shape}", flush=True)
            return v
        print(f"  cache stale {tag} (model/ids changed) -> re-embedding", flush=True)
    t0 = time.time()
    v = embed_all(texts, base, model, **kw)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(npy, v)
    json.dump({"model": model, "dims": int(v.shape[1]), "n": len(ids), "ids": ids}, open(meta, "w"))
    print(f"  embedded+cached {tag}: {v.shape} in {time.time()-t0:.0f}s", flush=True)
    return v


def load_label_vecs(distractor_dir, exclude_label, dim):
    """Load already-computed map embeddings as distractors, excluding one label."""
    chunks = []
    for f in sorted(glob.glob(os.path.join(distractor_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["embedding", "subset"])
        labs = [s.replace("Nemotron-Pretraining-Legal-", "") for s in t.column("subset").to_pylist()]
        v = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        v = v.astype(np.float32).reshape(len(t), -1)[:, :dim]
        keep = np.array([l != exclude_label for l in labs])
        if keep.any():
            chunks.append(v[keep])
    D = np.vstack(chunks)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return D


def metrics(ranks, ks=(1, 5, 10, 100)):
    r = np.asarray(ranks)
    m = {f"R@{k}": round(float(np.mean(r <= k)), 4) for k in ks}
    m["MRR@10"] = round(float(np.mean([1.0 / x if x <= 10 else 0.0 for x in r])), 4)
    m["median"] = float(np.median(r)); m["mean"] = round(float(np.mean(r)), 1)
    return m


def ranks_against(Q, gold_vecs, corpus_blocks):
    """rank = 1 + #docs scoring strictly above the gold doc (best-case under ties)."""
    gold_s = np.sum(Q * gold_vecs, axis=1)
    cnt = np.zeros(len(Q), dtype=np.int64)
    for D in corpus_blocks:
        for i in range(0, D.shape[0], 20000):                 # chunk to bound memory
            cnt += (Q @ D[i:i + 20000].T > gold_s[:, None]).sum(1)
    return cnt + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval")
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--split", default="eval", help='"eval" | "train" | "all"')
    ap.add_argument("--distractor-dir", default=None, help="dir of map parquet for open-corpus distractors")
    ap.add_argument("--exclude-label", default="eCFR-QA", help="label excluded from distractors (avoids leakage)")
    ap.add_argument("--max-chars", type=int, default=8000)
    a = ap.parse_args()
    cache_dir = os.path.join(a.eval_dir, "cache")

    qs_all = [json.loads(l) for l in open(os.path.join(a.eval_dir, "questions.jsonl"))]
    corpus = [json.loads(l) for l in open(os.path.join(a.eval_dir, "ecfr_corpus.jsonl"))]
    docpos = {c["doc_uuid"]: i for i, c in enumerate(corpus)}
    model = get_model(a.base_url)
    print(f"model: {model} | questions: {len(qs_all)} | corpus: {len(corpus)}", flush=True)

    # corpus embeddings (cached) — keyed by doc ids
    D = cached_embed("ecfr_corpus", [c["text"] for c in corpus], [c["doc_uuid"] for c in corpus],
                     cache_dir, a.base_url, model, max_chars=a.max_chars)
    dim = D.shape[1]

    distract = None
    if a.distractor_dir:
        distract = load_label_vecs(a.distractor_dir, a.exclude_label, dim)
        print(f"  open-corpus distractors: +{distract.shape[0]} docs (excluded '{a.exclude_label}')", flush=True)

    qs = qs_all if a.split == "all" else [q for q in qs_all if q.get("split") == a.split]
    gold = np.array([docpos[q["gold_uuid"]] for q in qs])
    print(f"evaluating split='{a.split}': {len(qs)} questions\n", flush=True)

    for variant, qtext in [
        ("with-instruction", [f"Instruct: {INSTRUCT}\nQuery: {q['question']}" for q in qs_all]),
        ("raw", [q["question"] for q in qs_all]),
    ]:
        Qall = cached_embed(f"queries.{variant}", qtext, [q["qid"] for q in qs_all],
                            cache_dir, a.base_url, model, max_chars=a.max_chars)
        sel = [i for i, q in enumerate(qs_all) if (a.split == "all" or q.get("split") == a.split)]
        Q = Qall[sel]
        gold_vecs = D[gold]
        r_closed = ranks_against(Q, gold_vecs, [D])
        print(f"[{variant} | closed eCFR-{len(corpus)//1000}k]  {metrics(r_closed)}", flush=True)
        if distract is not None:
            r_open = ranks_against(Q, gold_vecs, [D, distract])
            print(f"[{variant} | open  +{distract.shape[0]//1000}k distractors]  {metrics(r_open)}", flush=True)
        if variant == "with-instruction":
            out = [{"qid": q["qid"], "rank": int(r), "gold_title": q["gold_title"],
                    "gold_section": q["gold_section"], "question": q["question"]}
                   for q, r in zip(qs, r_closed)]
            json.dump(out, open(os.path.join(a.eval_dir, "baseline_ranks.json"), "w"))
        print(flush=True)


if __name__ == "__main__":
    main()
