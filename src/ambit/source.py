"""Streaming sources — yield fixed-size chunks of vectors so a 1M+ corpus is never
fully resident. Every ingestion path (a parquet file, a jsonl export, a numpy
memmap, or a model embedding raw items) is just an iterator of `Chunk`s, which the
one-pass `scan` consumes.

Parquet vectors are pulled via Arrow's zero-copy `flatten().to_numpy()` (not the
Python-object `to_pydict`), which is the dominant ingestion cost at scale.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .ingest import _EMB_KEYS, _ID_KEYS, _LABEL_KEYS, _pick


@dataclass
class Chunk:
    X: np.ndarray                       # (m, d)
    ids: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None


def expand(source):
    """A directory or glob pattern -> a sorted list of shard files; a single path
    or ndarray -> None (handled directly). Lets a 100k-embedding sharded dump be
    scanned as one source."""
    if not isinstance(source, (str, Path)):
        return None
    sp = str(source)
    if any(c in sp for c in "*?["):
        paths = sorted(glob.glob(sp))
    elif Path(sp).is_dir():
        d = Path(sp)
        paths = (sorted(glob.glob(str(d / "*.parquet"))) or sorted(glob.glob(str(d / "*.pq")))
                 or sorted(glob.glob(str(d / "*.npz"))) or sorted(glob.glob(str(d / "*.npy")))
                 or sorted(glob.glob(str(d / "*.jsonl"))))
    else:
        return None
    if not paths:
        raise FileNotFoundError(f"no parquet/npy/npz/jsonl files match {source!r}")
    return paths


def iter_chunks(source, *, embedding_col=None, id_col=None, label_col=None,
                batch_rows: int = 50_000) -> Iterator[Chunk]:
    if isinstance(source, np.ndarray):
        yield from _array_chunks(source, None, None, batch_rows)
        return
    shards = expand(source)
    if shards is not None:
        for fp in shards:                                 # sharded directory / glob
            yield from iter_chunks(fp, embedding_col=embedding_col, id_col=id_col,
                                   label_col=label_col, batch_rows=batch_rows)
        return
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"no such source: {p}")
    suf = p.suffix.lower()
    if suf == ".npy":
        yield from _array_chunks(np.load(p, mmap_mode="r"), None, None, batch_rows)  # streamed off disk
    elif suf == ".npz":
        z = np.load(p, allow_pickle=True)
        ek = _pick(_EMB_KEYS, z.files) or z.files[0]
        idk, lbk = _pick(_ID_KEYS, z.files), _pick(_LABEL_KEYS, z.files)
        yield from _array_chunks(z[ek], z[idk] if idk else None, z[lbk] if lbk else None, batch_rows)
    elif suf in (".parquet", ".pq"):
        yield from _parquet_chunks(p, embedding_col, id_col, label_col, batch_rows)
    elif suf in (".jsonl", ".ndjson"):
        yield from _jsonl_chunks(p, embedding_col, id_col, label_col, batch_rows)
    else:
        raise ValueError(f"unsupported streaming source {suf!r}; use .npy/.npz/.parquet/.jsonl")


def _array_chunks(X, ids, labels, batch):
    n = len(X)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        yield Chunk(np.asarray(X[s:e], dtype=np.float32),
                    None if ids is None else np.asarray(ids[s:e]),
                    None if labels is None else np.asarray(labels[s:e]))


def _arrow_vectors(col) -> np.ndarray:
    """A list<float> / fixed_size_list<float> Arrow column -> (m, d) float32, via the
    flattened child buffer (near zero-copy) instead of per-element Python objects."""
    import pyarrow as pa
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    m = len(col)
    flat = col.flatten().to_numpy(zero_copy_only=False)
    flat = np.asarray(flat, dtype=np.float32)
    if pa.types.is_fixed_size_list(col.type):
        d = col.type.list_size
    else:
        d = (len(flat) // m) if m else 0
    return np.ascontiguousarray(flat.reshape(m, d))


def _parquet_chunks(p, embedding_col, id_col, label_col, batch):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading parquet needs pyarrow: pip install 'ambit[io]'") from e
    pf = pq.ParquetFile(p)
    ec = embedding_col or _pick(_EMB_KEYS, pf.schema_arrow.names)
    if ec is None:
        raise ValueError(f"{p}: no embedding column found; pass embedding_col=")
    cols = [c for c in (ec, id_col, label_col) if c]
    for b in pf.iter_batches(batch_size=batch, columns=cols or None, use_threads=True):
        yield Chunk(
            _arrow_vectors(b.column(ec)),
            b.column(id_col).to_numpy(zero_copy_only=False) if id_col else None,
            b.column(label_col).to_numpy(zero_copy_only=False) if label_col else None,
        )


def _jsonl_chunks(p, embedding_col, id_col, label_col, batch):
    buf = []
    with open(p) as f:
        for ln in f:
            if ln.strip():
                buf.append(json.loads(ln))
            if len(buf) >= batch:
                yield _rows_to_chunk(buf, embedding_col, id_col, label_col)
                buf = []
    if buf:
        yield _rows_to_chunk(buf, embedding_col, id_col, label_col)


def _rows_to_chunk(rows, embedding_col, id_col, label_col):
    ec = embedding_col or _pick(_EMB_KEYS, rows[0])
    if ec is None:
        raise ValueError("no embedding field in jsonl; pass embedding_col=")
    return Chunk(np.asarray([r[ec] for r in rows], dtype=np.float32),
                 np.asarray([r.get(id_col) for r in rows]) if id_col else None,
                 np.asarray([r.get(label_col) for r in rows]) if label_col else None)
