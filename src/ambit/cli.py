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

from . import api
from .api import Diagnostics
from .config import Config


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
        mutual_knn=getattr(a, "mutual_knn", False),
        clusters=clusters,
        model=getattr(a, "model", None),
        base_url=getattr(a, "base_url", None),
        api_key=getattr(a, "api_key", None),
        embed_batch=getattr(a, "batch", 256),
        compare=getattr(a, "compare", None),
        compare_label=getattr(a, "compare_label", "B"),
        compare_id_col=getattr(a, "compare_id_col", None),
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


def _print_diagnostics(d: Diagnostics) -> None:
    """Format a Diagnostics result for the terminal (presentation only)."""
    sc = d.scan
    print(f"source                 {sc.source}")
    tag = f"  (~{sc.scanned:,} sampled, approx)" if sc.approximate else "  (full corpus, streamed)"
    print(f"items x dims           {sc.n:,} x {sc.dim}{tag}")
    print(f"mean L2 norm           {sc.norm_mean:.4f}  +/- {sc.norm_std:.4f}")
    print()
    print("--- resolution / isotropy ---")
    print(f"mean random-pair cos   {d.mean_cos:+.4f}   (isotropic ref ~ 0.000 +/- {d.isotropy_ref:.4f})")
    print(f"  verdict              {d.verdict}  (lower magnitude = higher resolution)")
    print(f"effective rank         {d.effective_rank:.1f} / {sc.dim}")
    print(f"participation ratio    {d.participation_ratio:.1f} / {sc.dim}")
    print(f"dims for 90% variance  {d.dims_for_90pct} / {sc.dim}")
    print()
    print("random-pair cosine distribution")
    print(_ascii_hist(d.cos, min(-0.2, float(d.cos.min())), max(0.8, float(d.cos.max()))))


def cmd_info(a) -> int:
    _print_diagnostics(api.diagnose(a.embeddings, config=_config(a)))
    return 0


def cmd_embed(a) -> int:
    n = api.embed(a.dataset, a.out, config=_config(a),
                  progress=lambda k: print(f"\r  embedded {k:,}", end="", flush=True))
    print(f"\nwrote {n:,} embeddings -> {a.out}")
    return 0


def cmd_report(a) -> int:
    rep = api.report(a.embeddings, out=a.out, config=_config(a))
    sc = rep.scan
    print(f"wrote {a.out}  ({sc.n:,} items x {sc.dim} dims, {rep.shown}/{rep.total} shown)")
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
    p.add_argument("--mutual-knn", action="store_true",
                   help="filter every kNN graph to reciprocal (mutual) neighbors — suppresses hubs")
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
    pr.add_argument("--compare", default=None,
                    help="second embeddings to diff against (same items, aligned by id) — enables the CMP figures")
    pr.add_argument("--compare-label", default="B", help="display name for the second set (primary is A)")
    pr.add_argument("--compare-id-col", default=None,
                    help="id column to align the two sets on (defaults to --id-col)")
    pr.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)
