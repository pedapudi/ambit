"""ambit CLI.

  ambit info   <embeddings>   one-pass streaming scan -> resolution diagnostics
  ambit embed  <dataset>      raw items -> vectors via an OpenAI-compatible endpoint
  ambit report <embeddings>   -> a self-contained, theme-adaptive HTML report

Every setting is a flag (or a `--config` JSON object); nothing is read from the
environment. Flags are gathered into a `Config` object that the run is described by.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import embed as embedmod
from . import metrics
from . import pipeline
from . import render
from .config import Config
from .config import enabled as fig_enabled
from .scan import scan as run_scan


def _config(a) -> Config:
    """Build the run Config from parsed args, then merge a --config JSON if given."""
    clusters = False if getattr(a, "no_cluster", False) else (
        a.clusters if getattr(a, "clusters", None) else "auto")
    cfg = Config(
        embedding_col=getattr(a, "embedding_col", None),
        id_col=getattr(a, "id_col", None),
        label_col=getattr(a, "label_col", None),
        text_col=getattr(a, "text_col", "text"),
        metric=getattr(a, "metric", "cosine"),
        sample=getattr(a, "sample", 20_000),
        pairs=getattr(a, "pairs", 200_000),
        batch_rows=getattr(a, "batch_rows", 50_000),
        approx=getattr(a, "approx", None),
        device=getattr(a, "device", "cpu"),
        projector=getattr(a, "projector", "pca"),
        k=getattr(a, "k", 10),
        knn_backend=getattr(a, "knn_backend", "auto"),
        clusters=clusters,
        model=getattr(a, "model", None),
        base_url=getattr(a, "base_url", None),
        api_key=getattr(a, "api_key", None),
        embed_batch=getattr(a, "batch", 256),
        title=getattr(a, "title", Config.title),
    )
    if getattr(a, "config", None):
        cfg = cfg.merge(json.load(open(a.config)))
    return cfg


def _ascii_hist(vals, lo, hi, bins: int = 30, width: int = 46) -> str:
    h, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    top = int(h.max()) or 1
    return "\n".join(f"  {edges[k]:+.2f}  {'#' * int(round(width * h[k] / top))}"
                     for k in range(bins))


def cmd_info(a) -> int:
    cfg = _config(a)
    sc = run_scan(a.embeddings, sample=cfg.sample, embedding_col=cfg.embedding_col,
                  id_col=cfg.id_col, label_col=cfg.label_col, metric=cfg.metric,
                  batch_rows=cfg.batch_rows, device=cfg.device, approx=cfg.approx)
    eigs = sc.eigs
    cos = metrics.random_pair_cosine(sc.sample.X, n_pairs=cfg.pairs)
    erank = metrics.effective_rank(eigs)
    pr = metrics.participation_ratio(eigs)
    d90 = metrics.dims_for_variance(eigs, 0.9)
    ref = metrics.isotropy_ref(sc.dim)
    m = float(cos.mean())
    print(f"source                 {sc.source}")
    tag = f"  (~{sc.scanned:,} sampled, approx)" if sc.approximate else "  (full corpus, streamed)"
    print(f"items x dims           {sc.n:,} x {sc.dim}{tag}")
    print(f"mean L2 norm           {sc.norm_mean:.4f}  +/- {sc.norm_std:.4f}")
    print()
    print("--- resolution / isotropy ---")
    print(f"mean random-pair cos   {m:+.4f}   (isotropic ref ~ 0.000 +/- {ref:.4f})")
    print(f"  verdict              {'anisotropic / crowded' if m > 4 * ref else 'near-isotropic'}"
          f"  (lower magnitude = higher resolution)")
    print(f"effective rank         {erank:.1f} / {sc.dim}")
    print(f"participation ratio    {pr:.1f} / {sc.dim}")
    print(f"dims for 90% variance  {d90} / {sc.dim}")
    print()
    print("random-pair cosine distribution")
    print(_ascii_hist(cos, min(-0.2, float(cos.min())), max(0.8, float(cos.max()))))
    return 0


def cmd_embed(a) -> int:
    cfg = _config(a)
    client = embedmod.EmbeddingClient(cfg.model, base_url=cfg.base_url,
                                      api_key=cfg.api_key, batch=cfg.embed_batch)
    n = embedmod.embed_dataset(a.dataset, a.out, client=client, text_col=cfg.text_col,
                               id_col=cfg.id_col, label_col=cfg.label_col, batch=cfg.embed_batch,
                               progress=lambda k: print(f"\r  embedded {k:,}", end="", flush=True))
    print(f"\nwrote {n:,} embeddings -> {a.out}")
    return 0


def cmd_report(a) -> int:
    cfg = _config(a)
    sc = run_scan(a.embeddings, sample=cfg.sample, embedding_col=cfg.embedding_col,
                  id_col=cfg.id_col, label_col=cfg.label_col, metric=cfg.metric,
                  batch_rows=cfg.batch_rows, device=cfg.device, approx=cfg.approx)
    ctx = pipeline.build_ctx(sc, projector=cfg.projector, pairs=cfg.pairs, k=cfg.k,
                             clusters=cfg.clusters, device=cfg.device, knn_backend=cfg.knn_backend)
    render.build_report(ctx, out=a.out, title=cfg.title, config=cfg)
    shown = sum(1 for key in render.FIGURES if fig_enabled(cfg.figures, key))
    print(f"wrote {a.out}  ({sc.n:,} items x {sc.dim} dims, {shown}/{len(render.FIGURES)} shown)")
    return 0


def _scan_args(p):
    p.add_argument("embeddings", help=".npy/.npz/.parquet/.jsonl")
    p.add_argument("--embedding-col", default=None)
    p.add_argument("--id-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--pairs", type=int, default=200_000)
    p.add_argument("--sample", type=int, default=20_000)
    p.add_argument("--batch-rows", type=int, default=50_000)
    p.add_argument("--k", type=int, default=10, help="neighbors for the kNN graph")
    p.add_argument("--device", default="cpu", help="cpu (numpy) | auto | cuda | mps | torch")
    p.add_argument("--approx", type=int, default=None,
                   help="cap the covariance/diagnostics to ~N rows (approximate, fast on 1M+)")
    p.add_argument("--knn-backend", default="auto",
                   choices=["auto", "pynndescent", "sklearn", "brute", "faiss"])
    p.add_argument("--config", default=None, help="JSON object overriding Config fields / figures")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ambit", description="Visualize how a dataset occupies an embedding space.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="streaming scan -> resolution diagnostics")
    _scan_args(pi)
    pi.set_defaults(func=cmd_info)

    pe = sub.add_parser("embed", help="embed raw items via an OpenAI-compatible endpoint")
    pe.add_argument("dataset", help=".jsonl/.csv/.parquet/.txt of raw items")
    pe.add_argument("--out", required=True, help="output .jsonl or .parquet")
    pe.add_argument("--model", required=True)
    pe.add_argument("--text-col", default="text")
    pe.add_argument("--id-col", default=None)
    pe.add_argument("--label-col", default=None)
    pe.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint base URL")
    pe.add_argument("--api-key", default=None, help="API key (omit for an unauthenticated endpoint)")
    pe.add_argument("--batch", type=int, default=256)
    pe.set_defaults(func=cmd_embed)

    pr = sub.add_parser("report", help="render a self-contained HTML occupancy report")
    _scan_args(pr)
    pr.add_argument("--out", default="ambit-report.html")
    pr.add_argument("--projector", default="pca", choices=["pca", "umap"])
    pr.add_argument("--title", default="ambit — embedding-space occupancy")
    pr.add_argument("--clusters", type=int, default=None, help="force k clusters for auto-labeling")
    pr.add_argument("--no-cluster", action="store_true", help="disable unsupervised labeling")
    pr.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)
