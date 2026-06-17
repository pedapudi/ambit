"""ambit's high-level library surface — the API the CLI (and any other caller) drives.

The package has always been factored as a pipeline of modules — `scan` streams a
corpus into a `Scan`, `pipeline.build_ctx` turns that into a `Ctx`, `render`
turns a `Ctx` into HTML — but the orchestration that wires them together used to
live inside the CLI. This module lifts that wiring into three library functions
that *return values* instead of printing, so ambit can be invoked from a notebook,
a web service, a batch job, or another program just as well as from the terminal:

    import ambit
    rep = ambit.report("embeddings.parquet", sample=50_000)
    rep.write("report.html")                       # or use rep.html / rep.ctx directly

    diag = ambit.diagnose("embeddings.parquet")
    print(diag.mean_cos, diag.verdict)             # numbers, not stdout text

    ambit.embed("items.jsonl", "vecs.parquet", model="text-embedding-3-small")

Each verb takes either a `Config` (the same object the CLI builds from its flags)
or plain keyword overrides; nothing is read from the environment. The CLI is now a
thin presentation layer: it builds a `Config`, calls one of these, and formats the
result for a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import embedding as _embed
from . import metrics
from . import pipeline
from . import render
from .config import Config, enabled
from .pipeline import Ctx
from .scan import Scan
from .scan import scan as _scan


def _as_config(config: Optional[Config], **overrides) -> Config:
    """Resolve the run Config: start from `config` (or the defaults) and apply any
    explicitly-passed keyword overrides. Every key in `overrides` was named by the
    caller, so all are applied; a key matching a `Config` field sets that field, and
    any other key is treated as a figure toggle (mirroring the `--config` JSON)."""
    cfg = config if isinstance(config, Config) else Config()
    return cfg.merge(overrides) if overrides else cfg


def _run_scan(source, cfg: Config) -> Scan:
    return _scan(source, sample=cfg.sample, embedding_col=cfg.embedding_col,
                 id_col=cfg.id_col, label_col=cfg.label_col, metric=cfg.metric,
                 batch_rows=cfg.batch_rows, device=cfg.device, approx=cfg.approx)


# --------------------------------------------------------------------------- info

@dataclass
class Diagnostics:
    """The result of a streaming scan plus the global resolution/isotropy metrics —
    the data behind `ambit info`, with no formatting attached."""
    scan: Scan
    cos: np.ndarray             # random-pair cosine sample over the reservoir
    mean_cos: float
    isotropy_ref: float         # expected sd of cosine under isotropy (1/sqrt(d))
    effective_rank: float
    participation_ratio: float
    dims_for_90pct: int

    @property
    def anisotropic(self) -> bool:
        """Same threshold the CLI verdict uses: mean |cos| well above the isotropic ref."""
        return self.mean_cos > 4 * self.isotropy_ref

    @property
    def verdict(self) -> str:
        return "anisotropic / crowded" if self.anisotropic else "near-isotropic"


def diagnose(source, *, config: Optional[Config] = None, **overrides) -> Diagnostics:
    """One streaming pass over `source` -> global resolution diagnostics (no I/O)."""
    cfg = _as_config(config, **overrides)
    sc = _run_scan(source, cfg)
    eigs = sc.eigs
    cos = metrics.random_pair_cosine(sc.sample.X, n_pairs=cfg.pairs)
    return Diagnostics(
        scan=sc, cos=cos, mean_cos=float(cos.mean()),
        isotropy_ref=metrics.isotropy_ref(sc.dim),
        effective_rank=metrics.effective_rank(eigs),
        participation_ratio=metrics.participation_ratio(eigs),
        dims_for_90pct=metrics.dims_for_variance(eigs, 0.9))


# ------------------------------------------------------------------------- report

@dataclass
class Report:
    """A rendered occupancy report plus everything it was built from, so a caller can
    write the HTML, inspect the `Ctx`, or re-render at a different figure set."""
    html: str
    ctx: Ctx
    config: Config
    shown: int                  # figures rendered under this config
    total: int                  # figures available

    @property
    def scan(self) -> Scan:
        return self.ctx.scan

    def write(self, path) -> str:
        """Write the HTML to `path` and return the path as a string."""
        Path(path).write_text(self.html, encoding="utf-8")
        return str(path)


def report(source, *, out=None, config: Optional[Config] = None, **overrides) -> Report:
    """Full pipeline: scan `source`, build the projection/kNN/cluster context, and
    render the self-contained HTML report. Writes to `out` when given; the HTML is
    also returned on the `Report` (`.html`) along with the `Ctx` (`.ctx`)."""
    cfg = _as_config(config, **overrides)
    sc = _run_scan(source, cfg)
    ctx = pipeline.build_ctx(sc, projector=cfg.projector, pairs=cfg.pairs, k=cfg.k,
                             clusters=cfg.clusters, device=cfg.device,
                             knn_backend=cfg.knn_backend)
    html = render.build_report(ctx, out=out, title=cfg.title, config=cfg)
    shown = sum(1 for key in render.FIGURES if enabled(cfg.figures, key))
    return Report(html=html, ctx=ctx, config=cfg, shown=shown, total=len(render.FIGURES))


def build_context(source, *, config: Optional[Config] = None, **overrides) -> Ctx:
    """Scan `source` and build the report `Ctx` without rendering — for callers that
    want the projected reservoir, kNN graph, clusters, and metrics as objects (e.g.
    to drive `local_anisotropy`, a custom figure, or an interactive view)."""
    cfg = _as_config(config, **overrides)
    sc = _run_scan(source, cfg)
    return pipeline.build_ctx(sc, projector=cfg.projector, pairs=cfg.pairs, k=cfg.k,
                              clusters=cfg.clusters, device=cfg.device,
                              knn_backend=cfg.knn_backend)


# -------------------------------------------------------------------------- embed

def embed(dataset, out, *, config: Optional[Config] = None, progress=None,
          **overrides) -> int:
    """Embed raw items from `dataset` into vectors at `out` via an OpenAI-compatible
    endpoint described by the Config (`model`, `base_url`, `api_key`, `embed_batch`).
    `progress` is an optional callback `(n_done: int) -> None`. Returns the count."""
    cfg = _as_config(config, **overrides)
    client = _embed.EmbeddingClient(cfg.model, base_url=cfg.base_url,
                                    api_key=cfg.api_key, batch=cfg.embed_batch)
    return _embed.embed_dataset(dataset, out, client=client, text_col=cfg.text_col,
                                id_col=cfg.id_col, label_col=cfg.label_col,
                                batch=cfg.embed_batch, progress=progress)
