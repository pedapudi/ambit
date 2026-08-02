#!/usr/bin/env python3
"""Embed one grid cell (corpus × encoder) → parquet map shards.

Reads a BEIR-layout corpus (corpus/*.parquet with _id/title/text) or a
jsonl.gz collection (ESCI: {"id"|"_id"|"docid", "title"?, "text"|"contents"}),
encodes with sentence-transformers (per-model document prefix from the
registry), and writes shards in the house schema (row_id, uuid, subset,
embedding) so ambit scan / measure_map.py work unchanged.

  python embed_cell.py --corpus ~/datasets/validation/trec-covid \
    --model ~/models/validation/BAAI-bge-large-en-v1.5 --model-key bge \
    --out ~/maps/validation/trec-covid/bge-large --device cuda:0
"""
import argparse, glob, gzip, json, os

import numpy as np

DOC_PREFIX = {
    "e5": "passage: ",
    "gemma": "title: none | text: ",
    "bge": "", "arctic": "", "minilm": "", "mpnet": "", "qwen3": "",
}


def iter_texts(corpus):
    pq_files = sorted(glob.glob(os.path.join(corpus, "corpus", "*.parquet")))
    if pq_files:
        import pyarrow.parquet as pq
        for f in pq_files:
            for b in pq.ParquetFile(f).iter_batches(batch_size=4096):
                ids = b.column("_id").to_pylist()
                titles = b.column("title").to_pylist() if "title" in b.schema.names else [""] * len(ids)
                texts = b.column("text").to_pylist()
                for i, t, x in zip(ids, titles, texts):
                    yield str(i), (f"{t}\n{x}" if t else str(x))
        return
    jl = glob.glob(os.path.join(corpus, "collection.jsonl.gz")) + \
        glob.glob(os.path.join(corpus, "*.jsonl.gz"))
    if not jl:
        raise SystemExit(f"no corpus found under {corpus}")
    with gzip.open(jl[0], "rt") as fh:
        for line in fh:
            d = json.loads(line)
            i = d.get("id") or d.get("_id") or d.get("docid")
            t = d.get("title") or ""
            x = d.get("text") or d.get("contents") or ""
            yield str(i), (f"{t}\n{x}" if t else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True, choices=sorted(DOC_PREFIX))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="cap on model.max_seq_length (some ST configs default "
                         "to 32k, which OOMs at batch size)")
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--rows-per-file", type=int, default=100_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(a.model, device=a.device,
                                model_kwargs={"torch_dtype": "bfloat16"})
    model.max_seq_length = min(model.max_seq_length or a.max_tokens, a.max_tokens)
    prefix = DOC_PREFIX[a.model_key]
    buf_u, buf_t, shard, total = [], [], 0, 0

    def flush():
        nonlocal shard, total
        if not buf_u:
            return
        out = os.path.join(a.out, f"map-{shard:05d}.parquet")
        if not os.path.exists(out):
            emb = model.encode([prefix + t for t in buf_t], batch_size=a.batch,
                               normalize_embeddings=True,
                               show_progress_bar=False).astype(np.float32)
            tbl = pa.table({
                "row_id": pa.array(range(total, total + len(buf_u)), pa.int64()),
                "uuid": pa.array(buf_u),
                "subset": pa.array([""] * len(buf_u)),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(emb.reshape(-1), pa.float32()), emb.shape[1]),
            }, metadata={"embedding_model": os.path.basename(a.model),
                         "embedding_dim": str(emb.shape[1])})
            tmp = out + ".tmp"
            pq.write_table(tbl, tmp)
            os.replace(tmp, out)
        total += len(buf_u)
        shard += 1
        print(f"shard {shard} done ({total} rows)", flush=True)
        buf_u.clear(); buf_t.clear()

    for u, t in iter_texts(a.corpus):
        buf_u.append(u)
        buf_t.append(t[:a.max_chars])
        if len(buf_u) >= a.rows_per_file:
            flush()
    flush()
    print(f"CELL EMBEDDED: {total} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
