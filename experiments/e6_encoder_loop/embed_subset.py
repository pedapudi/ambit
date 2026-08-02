#!/usr/bin/env python3
"""Re-embed the measurement subset with a served checkpoint, manifest order.

Reads the subset manifest and the staged text parquets (train + heldout
extractions), embeds every manifest row against an OpenAI-compatible
/v1/embeddings endpoint (documents raw — no instruction prefix, matching the
corpus pipeline), and writes an (n, d) float32 L2-normalized .npy aligned to
the manifest — the input measure_round.py expects.

Resumable: progress is checkpointed every --save-every chunks to <out>.part.npy
and continues from there on relaunch.

  python embed_subset.py --subset ~/e6-subset200k \
      --texts ~/e6-texts ~/e6-texts-heldout \
      --base-url http://localhost:8082/v1 --out ~/e6-rounds/r1.npy
"""
import argparse, glob, json, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import requests


def embed_batch(sess, base, model, texts, max_retry=6):
    payload = {"model": model, "input": [t if t.strip() else " " for t in texts]}
    for attempt in range(max_retry):
        try:
            r = sess.post(base + "/embeddings", json=payload, timeout=300)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code == 200:
            d = sorted(r.json()["data"], key=lambda x: x["index"])
            return [x["embedding"] for x in d]
        if r.status_code in (429, 500, 502, 503, 408):
            time.sleep(1.5 * (attempt + 1)); continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("retries exhausted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--texts", nargs="+", required=True,
                    help="dirs of <subset>/texts.parquet (train + heldout)")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=4)
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.subset, "manifest.json")))
    order = {u: i for i, u in enumerate(manifest["uuid"])}
    texts = {}
    for root in a.texts:
        for f in sorted(glob.glob(os.path.join(root, "*", "texts.parquet"))):
            t = pq.read_table(f)
            for u, x in zip(t.column("uuid").to_pylist(), t.column("text").to_pylist()):
                if u in order:
                    texts[u] = str(x)[:a.max_chars]
    missing = len(order) - len(texts)
    if missing:
        raise SystemExit(f"{missing} manifest uuids missing from text dirs")

    sess = requests.Session()
    model = sess.get(a.base_url + "/models", timeout=30).json()["data"][0]["id"]
    print(f"model: {model} | {len(order)} texts", flush=True)

    n = len(order)
    part = a.out + ".part.npy"
    state = a.out + ".part.json"
    if os.path.exists(part) and os.path.exists(state):
        V = np.load(part)
        done = json.load(open(state))["done"]
        print(f"resuming at row {done}", flush=True)
    else:
        V, done = None, 0

    all_texts = [None] * n
    for u, i in order.items():
        all_texts[i] = texts[u]

    t0, base_done = time.time(), done
    while done < n:
        end = min(done + a.chunk, n)
        chunk = all_texts[done:end]
        bs = [chunk[i:i + a.batch] for i in range(0, len(chunk), a.batch)]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            per = list(ex.map(lambda tb: embed_batch(sess, a.base_url, model, tb), bs))
        emb = np.asarray([v for vb in per for v in vb], dtype=np.float32)
        if V is None:
            V = np.zeros((n, emb.shape[1]), np.float32)
        V[done:end] = emb
        done = end
        if (done // a.chunk) % a.save_every == 0 or done == n:
            np.save(part, V)
            json.dump({"done": done}, open(state, "w"))
        rate = (done - base_done) / (time.time() - t0)
        print(f"{done}/{n} ({rate:.0f} rows/s)", flush=True)

    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    np.save(a.out, V)
    os.remove(part); os.remove(state)
    print(f"wrote {a.out} {V.shape}", flush=True)


if __name__ == "__main__":
    main()
