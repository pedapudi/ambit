"""RES 12 — Crowding skeleton (the minimum spanning tree, drawn where it lives).

The spatial companion to the crowding-pockets bars: the same merge-tree object,
drawn as geometry instead of summarized as prominence. Every entity is connected
into one tree by its shortest bridges (the native-space MST under cosine
distance); each edge is tinted by its **native** length — short edges (tight
bridges, the crowding skeleton) in the hot ramp, long edges (the roomy
background) faint. Members of the top pockets are ringed so the two figures
cross-reference. The layout is the 2-D projection, so an edge can fan across the
plot: the connection is real in the native space even when the projection
separates its endpoints — the same acknowledged limit as the kNN-edge overlay.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _box, _svg
from .. import crowding as cr


@figure
def fig_res_mst(ctx):
    W, H, PAD = 920, 560, 34
    PLOT_B = H - 60

    xy = getattr(ctx, "xy", None)
    es = ctx.es
    X = np.asarray(es.X)
    if xy is None or len(xy) < 64 or X.ndim != 2:
        note = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" fill="var(--ink-faint)" font-size="12" '
                f'text-anchor="middle">needs a projected reservoir</text>')
        return {"num": "RES 12", "order": 90.75, "name": "Crowding skeleton", "tech": "minimum spanning tree",
                "why": "No projected reservoir available to draw the tree over.",
                "svg": _svg(W, H, "Crowding skeleton (no data)", note),
                "legend": "", "reveal": "<b>Reveals:</b> where the tight bridges are.", "cls": ""}

    edges, idx = cr.spanning_edges(X, max_n=4096, seed=0)
    pk, pk_idx = cr.pockets(X, min_size=8, max_n=4096, seed=0)
    P = _box(np.asarray(xy, float)[idx], W, PLOT_B + PAD, pad=PAD)

    w = np.array([e[0] for e in edges])
    lo, hi = float(np.quantile(w, 0.02)), float(np.quantile(w, 0.98))
    rng_w = max(hi - lo, 1e-9)
    med = float(np.median(w))
    q10 = float(np.quantile(w, 0.10))

    # short (tight) edges hot and opaque; long (roomy) edges faint — draw long first
    def style(wv):
        t = float(np.clip((wv - lo) / rng_w, 0.0, 1.0))          # 0 = tightest
        if t < 0.25:
            pct = int(round(90 - 240 * t))                       # bad -> accent blend
            return f"color-mix(in srgb, var(--bad) {max(pct, 30)}%, var(--accent))", 1.6, 0.95
        return "var(--ink-faint)", 0.7, 0.28 + 0.25 * (1.0 - t)

    order = np.argsort(-w)                                        # longest first (drawn under)
    lines = []
    for e in order:
        wv, i, j = edges[e]
        col, sw, op = style(wv)
        lines.append(f'<line x1="{P[i,0]:.1f}" y1="{P[i,1]:.1f}" x2="{P[j,0]:.1f}" y2="{P[j,1]:.1f}" '
                     f'stroke="{col}" stroke-width="{sw}" stroke-opacity="{op:.2f}" '
                     f'vector-effect="non-scaling-stroke"/>')

    dots = "".join(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.4" '
                   f'fill="var(--ink-faint)" fill-opacity="0.55"/>' for i in range(len(P)))

    # ring the members of the top pockets (same subsample, same seed => same rows)
    pos_of = {int(r): k for k, r in enumerate(idx)}
    rings, labels = [], []
    for rank, p in enumerate(pk[:3]):
        pts = [pos_of[int(m)] for m in p["members"] if int(m) in pos_of]
        if not pts:
            continue
        for k in pts:
            rings.append(f'<circle cx="{P[k,0]:.1f}" cy="{P[k,1]:.1f}" r="3.4" fill="none" '
                         f'stroke="var(--bad)" stroke-width="0.9" stroke-opacity="0.85"/>')
        cx = float(np.mean([P[k, 0] for k in pts]))
        cy = float(np.mean([P[k, 1] for k in pts]))
        labels.append(f'<text x="{cx:.1f}" y="{cy - 12:.1f}" fill="var(--bad)" font-size="10.5" '
                      f'font-weight="700" text-anchor="middle" style="paint-order:stroke" '
                      f'stroke="var(--paper)" stroke-width="3">pocket · n={p["size"]}</text>')

    head = (f'<text x="{PAD}" y="26" fill="var(--ink-soft)" font-size="12">crowding skeleton · every entity '
            f'joined by its shortest bridges (native-space MST) · edge tint = native length</text>'
            f'<text x="{W-PAD}" y="26" fill="var(--ink-faint)" font-size="11" text-anchor="end" '
            f'style="font-variant-numeric:tabular-nums">{len(P):,} entities · median bridge {med:.2f} · '
            f'tightest decile ≤ {q10:.2f} (1−cos)</text>')
    foot = (f'<text x="{PAD}" y="{H-18}" fill="var(--ink-faint)" font-size="10">edges are native-space links '
            f'drawn on a projection — a long-looking hot edge is a tight bridge whose endpoints the '
            f'projection separated</text>')

    n_pk = len(pk)
    aria = (f"Crowding skeleton: the minimum spanning tree of {len(P)} entities under cosine distance, "
            f"drawn over the 2-D projection; short tight bridges are tinted hot, long background bridges "
            f"faint; members of the top {min(n_pk, 3)} pockets are ringed and labelled with their sizes; "
            f"median bridge length {med:.2f}.")
    return {
        "num": "RES 12", "order": 90.75, "name": "Crowding skeleton", "tech": "minimum spanning tree",
        "why": ("The merge tree drawn as geometry: every entity connected into one tree by its shortest "
                "native-space bridges. Runs of hot (short) edges are the crowding skeleton — the paths along "
                "which entities blur into each other; faint long edges are the roomy background. Ringed "
                "points are the members of the top pockets from the pockets figure — same object, seen "
                "spatially."),
        "svg": _svg(W, H, aria, head + '<g>' + "".join(lines) + '</g><g>' + dots + '</g><g>'
                    + "".join(rings) + "".join(labels) + '</g>' + foot),
        "legend": ('<span><i class="r"></i> tight bridge (short in native space)</span>'
                   '<span><i class="f"></i> roomy bridge</span>'
                   '<span><i class="r"></i> pocket member (ringed)</span>'),
        "reveal": ("<b>Reveals:</b> <b>where</b> the tight structure sits — whether the pockets are one "
                   "region or scattered, and which short-bridge runs will blur first as neighborhoods "
                   "tighten. Edges are native-space links; the projection only supplies the layout."),
        "cls": "",
    }
