"""ambit CLI.

  ambit info   <embeddings>   one-pass streaming scan -> resolution diagnostics
  ambit embed  <dataset>      raw items -> vectors via an OpenAI-compatible endpoint
  ambit report <embeddings>   -> a self-contained, theme-adaptive HTML report
"""

from __future__ import annotations

import argparse

import numpy as np

from . import embed as embedmod
from . import metrics
from . import pipeline
from . import render
from .scan import scan as run_scan


def _ascii_hist(vals, lo, hi, bins: int = 30, width: int = 46) -> str:
    h, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    top = int(h.max()) or 1
    return "\n".join(f"  {edges[k]:+.2f}  {'#' * int(round(width * h[k] / top))}"
                     for k in range(bins))


def cmd_info(a) -> int:
    sc = run_scan(a.embeddings, sample=a.sample, embedding_col=a.embedding_col,
                  id_col=a.id_col, label_col=a.label_col, metric=a.metric, batch_rows=a.batch_rows)
    eigs = sc.eigs
    cos = metrics.random_pair_cosine(sc.sample.X, n_pairs=a.pairs)
    erank = metrics.effective_rank(eigs)
    pr = metrics.participation_ratio(eigs)
    d90 = metrics.dims_for_variance(eigs, 0.9)
    ref = metrics.isotropy_ref(sc.dim)
    m = float(cos.mean())
    print(f"source                 {sc.source}")
    print(f"items x dims           {sc.n:,} x {sc.dim}  (full corpus, streamed)")
    print(f"mean L2 norm           {sc.norm_mean:.4f}  +/- {sc.norm_std:.4f}")
    print()
    print("--- resolution / isotropy ---")
    print(f"mean random-pair cos   {m:+.4f}   (isotropic ref ~ 0.000 +/- {ref:.4f})")
    print(f"  verdict              {'anisotropic / crowded' if m > 4 * ref else 'near-isotropic'}"
          f"  (lower magnitude = higher resolution)")
    print(f"effective rank         {erank:.1f} / {sc.dim}")
    print(f"participation ratio    {pr:.1f} / {sc.dim}")
    print(f"dims for 90% variance  {d90} / {sc.dim}")
    print(f"  (rank/variance exact over all {sc.n:,}; cosine over a {sc.sample.n:,}-vector reservoir)")
    print()
    print("random-pair cosine distribution")
    print(_ascii_hist(cos, min(-0.2, float(cos.min())), max(0.8, float(cos.max()))))
    return 0


def cmd_embed(a) -> int:
    client = embedmod.EmbeddingClient(a.model, base_url=a.base_url, batch=a.batch, env_key=a.api_key_env)
    n = embedmod.embed_dataset(a.dataset, a.out, client=client, text_col=a.text_col, id_col=a.id_col,
                               label_col=a.label_col, batch=a.batch,
                               progress=lambda k: print(f"\r  embedded {k:,}", end="", flush=True))
    print(f"\nwrote {n:,} embeddings -> {a.out}")
    return 0


def cmd_report(a) -> int:
    sc = run_scan(a.embeddings, sample=a.sample, embedding_col=a.embedding_col,
                  id_col=a.id_col, label_col=a.label_col, metric=a.metric, batch_rows=a.batch_rows)
    ctx = pipeline.build_ctx(sc, projector=a.projector, pairs=a.pairs)
    render.build_report(ctx, out=a.out, title=a.title)
    print(f"wrote {a.out}  ({sc.n:,} items x {sc.dim} dims, {len(render.FIGURES)} figures)")
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
    pe.add_argument("--base-url", default=None, help="default: $OPENAI_BASE_URL or api.openai.com/v1")
    pe.add_argument("--api-key-env", default="OPENAI_API_KEY")
    pe.add_argument("--batch", type=int, default=256)
    pe.set_defaults(func=cmd_embed)

    pr = sub.add_parser("report", help="render a self-contained HTML occupancy report")
    _scan_args(pr)
    pr.add_argument("--out", default="ambit-report.html")
    pr.add_argument("--projector", default="pca", choices=["pca", "umap"])
    pr.add_argument("--title", default="ambit — embedding-space occupancy")
    pr.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)
