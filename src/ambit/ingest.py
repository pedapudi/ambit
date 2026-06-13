"""Tier-0 ingestion: turn whatever embeddings the user has into an `EmbeddingSet`.

Supported sources (auto-detected by extension):
  .npy            a single (n, d) array
  .npz            an archive; the embedding array is found by key or by being the
                  largest 2-D array; ids/labels picked up from matching keys
  .parquet/.csv   a table with either a vector column (list/array per row) or a
                  wide numeric matrix; id/label/meta columns kept aside
  .jsonl          one JSON object per line with an embedding field + arbitrary meta
  np.ndarray      passed through directly

Only numpy is needed for .npy/.npz; tabular formats import pandas lazily so the
core install stays minimal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .types import EmbeddingSet

PathLike = Union[str, Path]

_EMB_KEYS = ("embeddings", "embedding", "X", "emb", "vectors", "vecs", "vector")
_ID_KEYS = ("ids", "id", "keys", "key")
_LABEL_KEYS = ("labels", "label", "y", "cluster")


def load(
    source: Union[PathLike, np.ndarray],
    *,
    embedding_col: Optional[str] = None,
    id_col: Optional[str] = None,
    label_col: Optional[str] = None,
    ids=None,
    labels=None,
    meta=None,
    metric: str = "cosine",
    normalize: bool = False,
) -> EmbeddingSet:
    """Load embeddings into an `EmbeddingSet`. Explicit `ids`/`labels`/`meta`
    (arrays or sidecar file paths) override anything carried by the file."""
    if isinstance(source, np.ndarray):
        es = EmbeddingSet(source, metric=metric, source="ndarray")
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"no such embeddings file: {p}")
        suf = p.suffix.lower()
        if suf == ".npy":
            es = EmbeddingSet(np.load(p), metric=metric, source=str(p))
        elif suf == ".npz":
            es = _from_npz(p, metric)
        elif suf in (".parquet", ".pq"):
            es = _from_table(p, "parquet", embedding_col, id_col, label_col, metric)
        elif suf == ".csv":
            es = _from_table(p, "csv", embedding_col, id_col, label_col, metric)
        elif suf in (".jsonl", ".ndjson"):
            es = _from_jsonl(p, embedding_col, id_col, label_col, metric)
        else:
            raise ValueError(
                f"unsupported embeddings format {suf!r}; use .npy/.npz/.parquet/.csv/.jsonl"
            )

    if ids is not None:
        es.ids = _resolve_sidecar(ids, es.n)
    if labels is not None:
        es.labels = _resolve_sidecar(labels, es.n)
    if meta is not None:
        es.meta = meta
    return es.normalize() if normalize else es


def _resolve_sidecar(val, n: int) -> np.ndarray:
    if isinstance(val, (str, Path)):
        p = Path(val)
        arr = np.load(p, allow_pickle=True) if p.suffix.lower() == ".npy" \
            else np.array([ln.rstrip("\n") for ln in open(p)])
    else:
        arr = np.asarray(val)
    if len(arr) != n:
        raise ValueError(f"sidecar length {len(arr)} != n_rows {n}")
    return arr


def _pick(keys, available) -> Optional[str]:
    for k in keys:
        if k in available:
            return k
    return None


def _from_npz(p: Path, metric: str) -> EmbeddingSet:
    z = np.load(p, allow_pickle=True)
    files = list(z.files)
    ek = _pick(_EMB_KEYS, files)
    if ek is None:  # fall back to the largest 2-D array in the archive
        two_d = [(f, z[f]) for f in files if getattr(z[f], "ndim", 0) == 2]
        if not two_d:
            raise ValueError(f"{p}: no embedding array found among keys {files}")
        ek = max(two_d, key=lambda kv: kv[1].size)[0]
    idk, lbk = _pick(_ID_KEYS, files), _pick(_LABEL_KEYS, files)
    return EmbeddingSet(z[ek], ids=z[idk] if idk else None,
                        labels=z[lbk] if lbk else None, metric=metric, source=str(p))


def _from_jsonl(p, embedding_col, id_col, label_col, metric) -> EmbeddingSet:
    rows = [json.loads(ln) for ln in open(p) if ln.strip()]
    if not rows:
        raise ValueError(f"{p}: empty jsonl")
    ec = embedding_col or _pick(_EMB_KEYS, rows[0])
    if ec is None:
        raise ValueError(f"{p}: no embedding field found; pass embedding_col=")
    X = np.asarray([r[ec] for r in rows], dtype=np.float32)
    ids = np.asarray([r.get(id_col) for r in rows]) if id_col else None
    labels = np.asarray([r.get(label_col) for r in rows]) if label_col else None
    meta = _meta_frame(rows, {ec, id_col, label_col})
    return EmbeddingSet(X, ids=ids, labels=labels, meta=meta, metric=metric, source=str(p))


def _from_table(p, kind, embedding_col, id_col, label_col, metric) -> EmbeddingSet:
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"reading {kind} needs pandas+pyarrow: pip install 'ambit[io]'") from e
    df = pd.read_parquet(p) if kind == "parquet" else pd.read_csv(p)
    ec = embedding_col or _pick(_EMB_KEYS, list(df.columns))
    if ec is not None and ec in df.columns and isinstance(df[ec].iloc[0], (list, tuple, np.ndarray)):
        X = np.asarray(np.vstack(df[ec].to_numpy()), dtype=np.float32)
        consumed = {ec, id_col, label_col}
    else:  # wide numeric matrix: every non-id/label numeric column is a dimension
        skip = {c for c in (id_col, label_col) if c}
        dim_cols = [c for c in df.columns
                    if c not in skip and np.issubdtype(df[c].dtype, np.number)]
        if not dim_cols:
            raise ValueError(f"{p}: no vector column or numeric matrix found; pass embedding_col=")
        X = df[dim_cols].to_numpy(dtype=np.float32)
        consumed = skip | set(dim_cols)
    ids = df[id_col].to_numpy() if id_col else None
    labels = df[label_col].to_numpy() if label_col else None
    leftover = [c for c in df.columns if c not in consumed]
    meta = df[leftover] if leftover else None
    return EmbeddingSet(X, ids=ids, labels=labels, meta=meta, metric=metric, source=str(p))


def _meta_frame(rows, drop):
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return None
    cols = [k for k in rows[0].keys() if k not in drop]
    return pd.DataFrame([{k: r.get(k) for k in cols} for r in rows]) if cols else None
