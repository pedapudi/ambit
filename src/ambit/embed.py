"""Tier-2: produce embeddings from raw items via an OpenAI-compatible endpoint.

`EmbeddingClient` speaks the plain `/v1/embeddings` HTTP contract with stdlib only,
so it works against OpenAI, vLLM, text-embeddings-inference, LM Studio, llama.cpp,
or any compatible server — configured by base_url + model, no provider SDK.

`embed_dataset` streams a raw dataset (jsonl/csv/parquet/txt) through the endpoint
in batches and writes vectors to .jsonl or .parquet, so a large corpus is embedded
without holding it all in memory.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np


class EmbeddingClient:
    def __init__(self, model: str, *, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 batch: int = 128, timeout: float = 60.0, max_retries: int = 5,
                 env_key: str = "OPENAI_API_KEY", env_base: str = "OPENAI_BASE_URL"):
        self.model = model
        self.base_url = (base_url or os.environ.get(env_base) or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get(env_key) or ""
        self.batch = batch
        self.timeout = timeout
        self.max_retries = max_retries

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        out = []
        for s in range(0, len(texts), self.batch):
            out.append(self._embed_batch(texts[s:s + self.batch]))
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)

    def _embed_batch(self, batch) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        url = self.base_url + "/embeddings"
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read())
                rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
                return np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"embeddings request to {url} failed after "
                                       f"{self.max_retries} tries: {e}") from e
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")


def embed_dataset(in_path, out_path, *, client: EmbeddingClient, text_col: str = "text",
                  id_col: Optional[str] = None, label_col: Optional[str] = None,
                  batch: int = 256, progress=None) -> int:
    out = Path(out_path)
    writer = (_ParquetEmbWriter(out) if out.suffix.lower() in (".parquet", ".pq")
              else _JsonlEmbWriter(out))
    n = 0
    try:
        for texts, ids, labels in _iter_text_batches(in_path, text_col, id_col, label_col, batch):
            writer.write(client.embed(texts), ids, labels, texts)
            n += len(texts)
            if progress:
                progress(n)
    finally:
        writer.close()
    return n


def _iter_text_batches(path, text_col, id_col, label_col, batch) -> Iterator[tuple]:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".jsonl", ".ndjson"):
        buf = []
        for ln in open(p):
            if ln.strip():
                buf.append(json.loads(ln))
            if len(buf) >= batch:
                yield _split(buf, text_col, id_col, label_col); buf = []
        if buf:
            yield _split(buf, text_col, id_col, label_col)
    elif suf == ".txt":
        buf = []
        for i, ln in enumerate(open(p)):
            buf.append({"text": ln.rstrip("\n"), "_i": i})
            if len(buf) >= batch:
                yield _split(buf, "text", "_i", None); buf = []
        if buf:
            yield _split(buf, "text", "_i", None)
    else:  # csv / parquet
        import pandas as pd
        df = pd.read_parquet(p) if suf in (".parquet", ".pq") else pd.read_csv(p)
        for s in range(0, len(df), batch):
            yield _split(df.iloc[s:s + batch].to_dict("records"), text_col, id_col, label_col)


def _split(rows, text_col, id_col, label_col):
    texts = [str(r[text_col]) for r in rows]
    ids = [r.get(id_col) for r in rows] if id_col else list(range(len(rows)))
    labels = [r.get(label_col) for r in rows] if label_col else [None] * len(rows)
    return texts, ids, labels


class _JsonlEmbWriter:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, V, ids, labels, texts):
        for k in range(len(texts)):
            rec = {"id": ids[k], "embedding": [float(x) for x in V[k]], "text": texts[k]}
            if labels[k] is not None:
                rec["label"] = labels[k]
            self.f.write(json.dumps(rec) + "\n")

    def close(self):
        self.f.close()


class _ParquetEmbWriter:
    def __init__(self, path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa, self.pq, self.path, self.writer = pa, pq, path, None

    def write(self, V, ids, labels, texts):
        cols = {"id": self.pa.array(ids),
                "embedding": self.pa.array([[float(x) for x in row] for row in V]),
                "text": self.pa.array([str(t) for t in texts])}
        if any(l is not None for l in labels):
            cols["label"] = self.pa.array(labels)
        tbl = self.pa.table(cols)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, tbl.schema)
        self.writer.write_table(tbl)

    def close(self):
        if self.writer is not None:
            self.writer.close()
