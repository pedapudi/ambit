#!/usr/bin/env python3
"""Truncation-matched preservation references for train_ambit.py.

The trainer embeds token-truncated views of each document; if its preservation
reference is the stored base embedding of the FULL text, the preservation term
becomes a text-length distillation instead of a model-drift control (the
round-1 failure). This script embeds the SAME truncated views with the BASE
model (via its serving endpoint) so reference and trainee see identical text.

Writes an (n_manifest, d) float32 npy with rows filled for the train split
(held-out rows stay zero — training never touches them). Pass to
train_ambit.py --ref-npy.

  python make_refs.py --subset ~/e6-subset200k --data ~/e6-texts \
      --model-dir ~/models/qwen3-embedding-0.6b \
      --base-url http://localhost:8081/v1 --max-tokens 512 \
      --out ~/e6-subset200k/ref_trunc512.npy
"""
import argparse, glob, json, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--data", required=True, help="train texts dir (subset/texts.parquet)")
    ap.add_argument("--model-dir", required=True, help="tokenizer source")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)

    manifest = json.load(open(os.path.join(a.subset, "manifest.json")))
    order = {u: i for i, u in enumerate(manifest["uuid"])}
    train = {u for u, s in zip(manifest["uuid"], manifest["split"]) if s == "train"}
    texts = {}
    for f in sorted(glob.glob(os.path.join(a.data, "*", "texts.parquet"))):
        t = pq.read_table(f)
        for u, x in zip(t.column("uuid").to_pylist(), t.column("text").to_pylist()):
            if u in train:
                texts[u] = str(x)[:a.max_chars]
    if len(texts) != len(train):
        raise SystemExit(f"texts cover {len(texts)}/{len(train)} train uuids")

    uuids = sorted(train, key=order.get)
    print(f"tokenize-truncating {len(uuids)} texts to {a.max_tokens} tokens", flush=True)
    trunc = []
    for u in uuids:
        ids = tok(texts[u], truncation=True, max_length=a.max_tokens)["input_ids"]
        trunc.append(tok.decode(ids, skip_special_tokens=True) or " ")

    sess = requests.Session()
    model = sess.get(a.base_url + "/models", timeout=30).json()["data"][0]["id"]

    def one(tb):
        for attempt in range(6):
            r = sess.post(a.base_url + "/embeddings",
                          json={"model": model, "input": tb}, timeout=300)
            if r.status_code == 200:
                return [x["embedding"] for x in
                        sorted(r.json()["data"], key=lambda d: d["index"])]
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

    t0 = time.time()
    bs = [trunc[i:i + a.batch] for i in range(0, len(trunc), a.batch)]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        per = list(ex.map(one, bs))
    emb = np.asarray([v for vb in per for v in vb], dtype=np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    print(f"embedded in {time.time()-t0:.0f}s", flush=True)

    out = np.zeros((len(order), emb.shape[1]), np.float32)
    for u, v in zip(uuids, emb):
        out[order[u]] = v
    np.save(a.out, out)
    print(f"wrote {a.out} {out.shape}", flush=True)


if __name__ == "__main__":
    main()
