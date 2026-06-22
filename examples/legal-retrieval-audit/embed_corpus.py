#!/usr/bin/env python3
"""Embed ONE Nemotron legal subset via the vLLM /v1/embeddings endpoint, truncate
to --dims (Matryoshka) + renormalize, and write parquet chunks matching the
existing pipeline schema (row_id, uuid, subset, embedding). Memory-safe (streams;
holds at most one chunk of text) and resumable (skips existing chunk files)."""
import argparse, glob, os, random, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, requests


def detect_model(sess, base):
    return sess.get(f"{base}/models", timeout=30).json()["data"][0]["id"]


def embed_batch(sess, base, model, texts, max_retry=6, backoff=1.0):
    url = f"{base}/embeddings"
    payload = {"model": model, "input": [t if t.strip() else " " for t in texts]}
    attempt = 0
    while True:
        try:
            r = sess.post(url, json=payload, timeout=300)
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retry:
                raise RuntimeError(f"network error after {attempt}: {e}")
            time.sleep(backoff * 2 ** attempt); continue
        if r.status_code == 200:
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        body = (r.text or "")[:200]; low = body.lower()
        if r.status_code == 400 and ("context" in low or "maximum" in low or "token" in low):
            if len(texts) > 1:
                h = len(texts) // 2
                return (embed_batch(sess, base, model, texts[:h], max_retry, backoff)
                        + embed_batch(sess, base, model, texts[h:], max_retry, backoff))
            payload["input"] = [texts[0][: max(200, len(texts[0]) // 2)]]; continue
        if r.status_code in (429, 500, 502, 503, 408):
            attempt += 1
            if attempt > max_retry:
                raise RuntimeError(f"HTTP {r.status_code}: {body}")
            time.sleep(backoff * 2 ** attempt); continue
        raise RuntimeError(f"HTTP {r.status_code}: {body}")


def embed_chunk(sess, base, model, texts, dims, batch, workers):
    bs = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        per = list(ex.map(lambda tb: embed_batch(sess, base, model, tb), bs))
    emb = np.asarray([v for vb in per for v in vb], dtype=np.float32)
    if dims and emb.shape[1] > dims:                       # Matryoshka truncate + renorm
        emb = emb[:, :dims]
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/datasets/Nemotron-Pretraining-Legal-v1"))
    ap.add_argument("--subset", required=True, help="subset dir name (e.g. Nemotron-Pretraining-Legal-eCFR-QA)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--target", type=int, default=500000)
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rows-per-file", type=int, default=20000)
    ap.add_argument("--row-id-offset", type=int, default=500000)
    ap.add_argument("--prefix", default="emb", help="chunk filename prefix (one per subset for a shared dir)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.data, a.subset, "*.parquet")))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    target = min(a.target, total)
    rng = random.Random(a.seed)
    keep = set(rng.sample(range(total), target))           # deterministic random row indices
    print(f"{a.subset}: {total} rows -> sampling {target}", flush=True)

    sess = requests.Session()
    model = detect_model(sess, a.base_url)
    print(f"model: {model} | dims out: {a.dims}", flush=True)

    buf_t, buf_u = [], []          # text, uuid for the current chunk
    gidx = 0                        # global row counter over the subset
    kept = 0; chunk = 0; t0 = time.time()

    def flush():
        nonlocal chunk, kept
        if not buf_t:
            return
        out = os.path.join(a.out_dir, f"{a.prefix}-{chunk:06d}.parquet")
        if not os.path.exists(out):
            emb = embed_chunk(sess, a.base_url, model, buf_t, a.dims, a.batch, a.workers)
            dim = emb.shape[1]
            n = len(buf_t)
            rid = list(range(a.row_id_offset + kept, a.row_id_offset + kept + n))
            meta = {"embedding_model": model, "embedding_dim": str(dim), "embedding_dtype": "float32"}
            if a.dims and dim < 1024:
                meta["matryoshka_from"] = "1024"
            tbl = pa.table({
                "row_id": pa.array(rid, type=pa.int64()),
                "uuid": pa.array(buf_u),
                "subset": pa.array([a.subset] * n),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1), type=pa.float32()), dim),
            }, metadata=meta)
            tmp = out + ".tmp"; pq.write_table(tbl, tmp); os.replace(tmp, out)
            print(f"  wrote {out} ({kept + n}/{target}, {(kept+n)/(time.time()-t0):.0f} rows/s)", flush=True)
        kept += len(buf_t); chunk += 1
        buf_t.clear(); buf_u.clear()

    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text", "uuid"]):
            ts = b.column("text").to_pylist(); us = b.column("uuid").to_pylist()
            for t, u in zip(ts, us):
                if gidx in keep:
                    buf_t.append(str(t)[:a.max_chars]); buf_u.append(u)
                    if len(buf_t) >= a.rows_per_file:
                        flush()
                gidx += 1
    flush()
    print(f"done: {kept} rows -> {a.out_dir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
