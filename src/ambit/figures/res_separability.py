"""Label-aware separability panel — does the partition occupy distinct regions of the
space? Draws the per-label geometry — the centroid-cosine matrix and kNN purity that
carry the interpretation. The partition is the provided labels, or — when none are given — ambit's own
clusters; in the unsupervised case the panel also reports cluster **stability**
(bootstrap ARI) and the number of **modes** (graph-Laplacian eigengap), and badges the
read as geometric, not semantic. All numbers come from `separability.compute`.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg, _box
from .. import separability as sep


def _trunc(s, n: int = 15) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _placeholder(ctx, note: str):
    W, H = 760, 470
    P = _box(ctx.xy, W, H)
    dots = "".join(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.2" '
                   f'fill="var(--ink-faint)" fill-opacity="0.4"/>' for i in range(len(P)))
    body = dots + (f'<text x="{W/2:.0f}" y="{H/2:.0f}" text-anchor="middle" font-size="12" '
                   f'fill="var(--ink-faint)">{note}</text>')
    return {"num": "RES 05b", "order": 94.5, "name": "Separability panel", "tech": "centroid · purity",
            "why": "Whether the partition occupies distinct regions: a centroid-cosine matrix, kNN "
                   "purity, and — for discovered clusters — stability and modal structure.",
            "svg": _svg(W, H, f"Separability panel unavailable: {note}.", body),
            "legend": f'<span><i class="f"></i> {note}</span>',
            "reveal": f"<b>Reveals:</b> {note}.", "cls": ""}


@figure
def fig_res_separability(ctx):
    labels = ctx.labels
    if labels is None:
        return _placeholder(ctx, "needs cluster labels")
    supervised = (ctx.labels_source == "provided")
    S = sep.compute(ctx.es.X, labels, ctx.knn_idx, supervised=supervised)
    if S is None:
        return _placeholder(ctx, "needs ≥2 groups")

    groups, counts, C, pur = list(S.groups), list(S.counts), S.centroids_cos, S.purity_per_group

    # cap groups for legibility (keep the largest, preserve order)
    CAP = 14
    capped = len(groups) > CAP
    if capped:
        idx = np.sort(np.argsort(counts)[::-1][:CAP])
        groups = [groups[i] for i in idx]
        counts = [counts[i] for i in idx]
        C = C[np.ix_(idx, idx)]
        pur = None if pur is None else pur[idx]
    g = len(groups)
    src = "provided" if supervised else (ctx.labels_source or "clustered")

    W, H = 760, 470
    cell = int(max(16, min(40, 300 // g)))
    hx0, hy0 = 92, 96
    hs = cell * g
    body = []

    # ---- header subtitle + (unsupervised) badge ---------------------------------
    body.append(f'<text x="{hx0}" y="26" fill="var(--ink-soft)" font-size="12">'
                f'{g}{"+" if capped else ""} groups · {src} · separability'
                f'{" (top 14 shown)" if capped else ""}</text>')
    if not supervised:
        stab = "—" if S.stability is None else f"{S.stability:.2f}"
        modes = "—" if S.n_modes is None else f"{S.n_modes}"
        body.append(f'<text x="{W-20}" y="22" fill="var(--caution)" font-size="11" font-weight="700" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">'
                    f'unsupervised · stability {stab} · modes ≈ {modes}</text>')
        body.append(f'<text x="{W-20}" y="36" fill="var(--ink-faint)" font-size="9" text-anchor="end">'
                    f'geometric separability of discovered clusters, not semantic</text>')

    # ---- centroid-cosine heatmap (accent ramp; diagonal blanked) ----------------
    body.append(f'<text x="{hx0}" y="{hy0-22}" fill="var(--ink-faint)" font-size="10">'
                f'centroid cosine — deeper = two groups share a direction (entangled)</text>')
    show_text = cell >= 26 and g <= 8
    maxoff, mi, mj = -2.0, 0, 0
    for i in range(g):
        for j in range(g):
            px, py = hx0 + j * cell, hy0 + i * cell
            if i == j:
                body.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell}" height="{cell}" '
                            f'fill="none" stroke="var(--rule-soft)" stroke-width="0.6" '
                            f'shape-rendering="crispEdges"/>')
                continue
            v = float(C[i, j])
            if v > maxoff:
                maxoff, mi, mj = v, i, j
            pct = int(round(min(max(v, 0.0), 1.0) * 100))
            fill = f'color-mix(in srgb, var(--accent) {pct}%, transparent)'
            body.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell}" height="{cell}" '
                        f'fill="{fill}" shape-rendering="crispEdges"/>')
            if show_text:
                body.append(f'<text x="{px+cell/2:.1f}" y="{py+cell/2+3:.1f}" fill="var(--ink-soft)" '
                            f'font-size="8.5" text-anchor="middle" '
                            f'style="font-variant-numeric:tabular-nums">{v:.2f}</text>')
    body.append(f'<rect x="{hx0}" y="{hy0}" width="{hs}" height="{hs}" fill="none" '
                f'stroke="var(--rule)" stroke-width="1" shape-rendering="crispEdges"/>')
    # index ticks (top + left); the purity bars carry the index→name key
    for k in range(g):
        body.append(f'<text x="{hx0+(k+0.5)*cell:.1f}" y="{hy0-5}" fill="var(--ink-faint)" '
                    f'font-size="8.5" text-anchor="middle" style="font-variant-numeric:tabular-nums">{k}</text>')
        body.append(f'<text x="{hx0-5}" y="{hy0+(k+0.5)*cell+3:.1f}" fill="var(--ink-faint)" '
                    f'font-size="8.5" text-anchor="end" style="font-variant-numeric:tabular-nums">{k}</text>')

    # ---- per-group purity bars (rows aligned with the heatmap) ------------------
    bx0, bx1 = 474, 724
    track = bx1 - bx0
    body.append(f'<text x="{bx0}" y="{hy0-22}" fill="var(--ink-faint)" font-size="10">'
                f'kNN purity per group — share of each item\'s neighbors with its own label</text>')
    if pur is None:
        body.append(f'<text x="{(bx0+bx1)/2:.0f}" y="{hy0+hs/2:.0f}" text-anchor="middle" '
                    f'font-size="11" fill="var(--ink-faint)">purity needs a kNN backend</text>')
    else:
        # overall purity reference rule
        ov = float(S.purity_overall or 0.0)
        ovx = bx0 + ov * track
        body.append(f'<line x1="{ovx:.1f}" y1="{hy0-4}" x2="{ovx:.1f}" y2="{hy0+hs+4}" '
                    f'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="2 3" opacity="0.8"/>')
        body.append(f'<text x="{ovx:.1f}" y="{hy0+hs+16}" fill="var(--ink-faint)" font-size="8.5" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">overall {ov:.2f}</text>')
        bh = max(4.0, cell * 0.62)
        for i in range(g):
            cy = hy0 + (i + 0.5) * cell
            val = float(pur[i])
            body.append(f'<text x="{bx0-8:.1f}" y="{cy+3:.1f}" fill="var(--ink-soft)" font-size="9" '
                        f'text-anchor="end">[{i}] {_trunc(groups[i])}</text>')
            body.append(f'<rect x="{bx0:.1f}" y="{cy-bh/2:.1f}" width="{track}" height="{bh:.1f}" '
                        f'fill="var(--rule-soft)" opacity="0.5"/>')
            body.append(f'<rect x="{bx0:.1f}" y="{cy-bh/2:.1f}" width="{max(1.0,val*track):.1f}" '
                        f'height="{bh:.1f}" fill="var(--accent)"/>')
            body.append(f'<text x="{bx0+val*track+5:.1f}" y="{cy+3:.1f}" fill="var(--ink-faint)" '
                        f'font-size="8.5" style="font-variant-numeric:tabular-nums">{val:.2f}</text>')

    # ---- scorecard line ---------------------------------------------------------
    sy = hy0 + hs + 44
    ov_txt = "—" if S.purity_overall is None else f"{S.purity_overall:.2f}"
    parts = [("silhouette", f"{S.silhouette:+.2f}"), ("Fisher ratio", f"{S.fisher:.2f}"),
             ("kNN purity", ov_txt)]
    sx = hx0
    for label, val in parts:
        body.append(f'<text x="{sx}" y="{sy}" fill="var(--ink-faint)" font-size="10">{label}</text>')
        body.append(f'<text x="{sx}" y="{sy+16}" fill="var(--accent)" font-size="14" font-weight="700" '
                    f'style="font-variant-numeric:tabular-nums">{val}</text>')
        sx += 150

    aria = (f"Separability panel over {g} {src} groups: a centroid-cosine matrix (deeper cells mark "
            f"groups whose mean directions align, i.e. collapse together), per-group kNN purity bars, "
            f"and a scorecard — silhouette {S.silhouette:+.2f}, Fisher ratio {S.fisher:.2f}, overall "
            f"kNN purity {ov_txt}." + ("" if supervised else
            f" The partition is ambit's own clusters; cluster stability (ARI) and the number of modes "
            f"qualify how much to trust it."))

    return {
        "num": "RES 05b", "order": 94.5,
        "name": "Separability panel", "tech": "centroid · purity · silhouette",
        "why": f"Whether the {src} partition occupies distinct regions of the space. The centroid-cosine "
               f"matrix shows which groups collapse together (most-entangled pair {mi}↔{mj} at "
               f"cos {maxoff:+.2f}); the bars show per-group kNN purity (overall {ov_txt}); the scorecard "
               f"adds silhouette and Fisher ratio." + ("" if supervised else
               " These are ambit's discovered clusters — stability and modes say whether that structure "
               "is worth interpreting, and the read is geometric, not semantic."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="a"></i> centroid cosine — deeper = more entangled</span>'
                  '<span><i class="a"></i> kNN purity per group (bar)</span>'
                  '<span><i class="dash"></i> overall purity</span>'
                  + ('' if supervised else '<span><i class="c"></i> unsupervised — geometric, not semantic</span>'),
        "reveal": "<b>Reveals:</b> whether your groups are geometrically distinct — light off-diagonal "
                  "cells and long purity bars mean the partition lives in separate regions; a deep "
                  "off-diagonal cell is two groups sharing a direction (collapsed together).",
        "cls": "",
    }
