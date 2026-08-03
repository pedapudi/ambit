#!/usr/bin/env python3
"""Score one grid cell against its qrels: nDCG@10, recall@100, and the
per-relevant-document failure labels for H2.

Run AFTER the cell's readouts are frozen. Embeds the queries with the same
encoder (per-model query prefix), exact search over the cell map on GPU,
scores against BEIR/TREC qrels (tsv: query-id, corpus-id, score — header
optional; or .qrels: qid 0 docid rel).

H2 outcome per relevant doc (rel >= --rel-min): "fails at its own queries"
= for at least one query where it is relevant, more than --fail-rank other
documents outrank it.

  python score_cell.py --map ~/maps/validation/trec-covid/bge-large \
    --queries ~/datasets/validation/trec-covid/queries \
    --qrels ~/datasets/validation/trec-covid-qrels/test.tsv \
    --model ~/models/validation/BAAI-bge-large-en-v1.5 --model-key bge \
    --out ~/validation-runs/trec-covid.bge-large.score.json
"""
import argparse, glob, json, math, os

import numpy as np

QUERY_PREFIX = {
    "e5": "query: ",
    "gemma": "task: search result | query: ",
    "bge": "Represent this sentence for searching relevant passages: ",
    "arctic": "Represent this sentence for searching relevant passages: ",
    "minilm": "", "mpnet": "",
    "qwen3": ("Instruct: Given a web search query, retrieve relevant passages "
              "that answer the query\nQuery: "),
}


def load_map(map_dir):
    import pyarrow.parquet as pq
    us, vs = [], []
    for f in sorted(glob.glob(os.path.join(map_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["uuid", "embedding"])
        us.extend(t.column("uuid").to_pylist())
        v = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        vs.append(v.astype(np.float32).reshape(len(t), -1))
    return us, np.vstack(vs)


def load_queries(qpath):
    if qpath.endswith((".jsonl", ".jsonl.gz")):        # e.g. ESCI dev-us.jsonl.gz
        import gzip
        op = gzip.open if qpath.endswith(".gz") else open
        ids, texts = [], []
        with op(qpath, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                ids.append(str(d.get("query_id") or d.get("qid") or
                               d.get("_id") or d.get("id")))
                texts.append(str(d.get("query") or d.get("text")))
        return ids, texts
    files = sorted(glob.glob(os.path.join(qpath, "*.parquet"))) or \
        sorted(glob.glob(os.path.join(qpath, "queries", "*.parquet")))
    import pyarrow.parquet as pq
    ids, texts = [], []
    for f in files:
        t = pq.read_table(f)
        ids.extend(str(x) for x in t.column("_id").to_pylist())
        texts.extend(str(x) for x in t.column("text").to_pylist())
    return ids, texts


def load_qrels(path):
    rels = {}
    with open(path) as fh:
        for line in fh:
            parts = line.replace("\t", " ").split()
            if not parts or parts[0].lower() in ("query-id", "qid"):
                continue
            if len(parts) == 4:                       # TREC: qid 0 docid rel
                q, _, d, r = parts
            else:                                     # BEIR tsv: qid docid rel
                q, d, r = parts[:3]
            rels.setdefault(str(q), {})[str(d)] = int(float(r))
    return rels


def ndcg_at_k(ranked_rels, ideal_rels, k=10):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(ranked_rels[:k]))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True, choices=sorted(QUERY_PREFIX))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rel-min", type=int, default=1)
    ap.add_argument("--fail-rank", type=int, default=10)
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    uuids, X = load_map(a.map)
    pos = {u: i for i, u in enumerate(uuids)}
    qids, qtexts = load_queries(a.queries)
    rels = load_qrels(a.qrels)
    keep = [i for i, q in enumerate(qids) if q in rels]
    qids = [qids[i] for i in keep]
    qtexts = [qtexts[i] for i in keep]
    print(f"map {X.shape} | {len(qids)} judged queries", flush=True)

    model = SentenceTransformer(a.model, device=a.device,
                                model_kwargs={"torch_dtype": "bfloat16"})
    Q = model.encode([QUERY_PREFIX[a.model_key] + t for t in qtexts],
                     batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)

    Xt = torch.tensor(X, device=a.device)
    Qt = torch.tensor(Q, device=a.device)
    K = 1000
    top_idx = torch.zeros(len(Q), K, dtype=torch.int64)
    for s in range(0, len(Q), 256):
        S = Qt[s:s + 256] @ Xt.T
        top_idx[s:s + 256] = torch.topk(S, min(K, len(X)), dim=1).indices.cpu()
    top_idx = top_idx.numpy()

    ndcgs, recalls = [], []
    doc_fail, doc_ok = set(), set()
    for i, q in enumerate(qids):
        r = rels[q]
        rel_docs = {d for d, s in r.items() if s >= a.rel_min}
        if not rel_docs:
            continue
        ranked = [uuids[j] for j in top_idx[i]]
        ranked_rels = [r.get(d, 0) for d in ranked]
        ideal = sorted(r.values(), reverse=True)
        ndcgs.append(ndcg_at_k(ranked_rels, ideal, 10))
        top100 = set(ranked[:100])
        recalls.append(len(rel_docs & top100) / len(rel_docs))
        # H2 outcome: relevant doc outranked beyond fail_rank at this query
        rank_of = {d: j + 1 for j, d in enumerate(ranked)}
        for d in rel_docs:
            if d in pos:
                (doc_fail if rank_of.get(d, K + 1) > a.fail_rank else doc_ok).add(d)
    doc_ok -= doc_fail

    out = {"map": a.map, "n_queries": len(ndcgs),
           "ndcg@10": float(np.mean(ndcgs)),
           "recall@100": float(np.mean(recalls)),
           "h2": {"fail_docs": sorted(doc_fail), "ok_docs": sorted(doc_ok)}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "h2"}, indent=1),
          f'| h2 docs: {len(doc_fail)} fail / {len(doc_ok)} ok', flush=True)


if __name__ == "__main__":
    main()
