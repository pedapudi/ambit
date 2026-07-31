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
    P[:, 1] += 16                                       # clear the two-line header

    w = np.array([e[0] for e in edges])
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med))) * 1.4826
    cut = med - 3.0 * max(mad, 1e-9)

    # Only bridges tighter than the bulk are drawn — the long "roomy" edges are
    # the hairball, and the empty space already says roomy. Robust cut: median
    # − 3 MAD-sigmas, capped so the figure can never re-busy itself.
    hot = np.flatnonzero(w <= cut)
    hot = hot[np.argsort(w[hot])][:600]
    lo = float(w[hot].min()) if hot.size else cut
    rng_w = max(cut - lo, 1e-9)
    lines = []
    for e in hot[::-1]:                                           # tightest drawn last (on top)
        wv, i, j = edges[e]
        t = float(np.clip((wv - lo) / rng_w, 0.0, 1.0))           # 0 = tightest
        pct = int(round(95 - 60 * t))
        lines.append(f'<line x1="{P[i,0]:.1f}" y1="{P[i,1]:.1f}" x2="{P[j,0]:.1f}" y2="{P[j,1]:.1f}" '
                     f'stroke="color-mix(in srgb, var(--bad) {pct}%, var(--accent))" '
                     f'stroke-width="1.7" stroke-opacity="0.9" vector-effect="non-scaling-stroke"/>')

    dots = "".join(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.4" '
                   f'fill="var(--ink-faint)" fill-opacity="0.55"/>' for i in range(len(P)))

    # ring the members of the top pockets (same subsample, same seed => same rows)
    pos_of = {int(r): k for k, r in enumerate(idx)}
    rings, lab_pos = [], []
    for rank, p in enumerate(pk[:3]):
        pts = [pos_of[int(m)] for m in p["members"] if int(m) in pos_of]
        if not pts:
            continue
        for k in pts:
            rings.append(f'<circle cx="{P[k,0]:.1f}" cy="{P[k,1]:.1f}" r="3.4" fill="none" '
                         f'stroke="var(--bad)" stroke-width="0.9" stroke-opacity="0.85"/>')
        cx = float(np.mean([P[k, 0] for k in pts]))
        cy = float(np.mean([P[k, 1] for k in pts]))
        lab_pos.append([cx, cy - 12.0, p["size"]])
    # de-overlap: nearby labels are pushed apart vertically until they clear
    lab_pos.sort(key=lambda a: a[1])
    for i in range(1, len(lab_pos)):
        if abs(lab_pos[i][0] - lab_pos[i - 1][0]) < 110 and \
           lab_pos[i][1] - lab_pos[i - 1][1] < 14:
            lab_pos[i][1] = lab_pos[i - 1][1] + 14
    labels = [f'<text x="{cx:.1f}" y="{ly:.1f}" fill="var(--bad)" font-size="10.5" '
              f'font-weight="700" text-anchor="middle" style="paint-order:stroke" '
              f'stroke="var(--paper)" stroke-width="3">pocket · n={sz}</text>'
              for cx, ly, sz in lab_pos]

    head = (f'<text x="{PAD}" y="24" fill="var(--ink-soft)" font-size="12">crowding skeleton · only the '
            f'bridges tighter than the bulk are drawn</text>'
            f'<text x="{PAD}" y="42" fill="var(--ink-faint)" font-size="10" '
            f'style="font-variant-numeric:tabular-nums">{len(lines):,} tight bridges (of {len(edges):,} · '
            f'cut = median − 3 robust sd = {cut:.2f}) · {len(P):,} entities · median bridge {med:.2f} (1−cos)</text>')
    foot = (f'<text x="{PAD}" y="{H-18}" fill="var(--ink-faint)" font-size="10">bridges are native-space '
            f'links on a projected layout — a long-looking one is a tight bridge the projection '
            f'stretched</text>')
    if not lines:
        head += (f'<text x="{W/2:.0f}" y="{(PLOT_B+PAD)/2:.0f}" fill="var(--good)" font-size="13" '
                 f'text-anchor="middle">no bridge is tighter than the bulk — no crowding skeleton '
                 f'to draw</text>')

    n_pk = len(pk)
    aria = (f"Crowding skeleton: of the {len(edges)} minimum-spanning-tree bridges over {len(P)} "
            f"entities, only the {len(lines)} tighter than the bulk (below {cut:.2f}) are drawn, "
            f"tinted by tightness, over the 2-D projection; members of the top {min(n_pk, 3)} pockets "
            f"are ringed and labelled with their sizes; median bridge length {med:.2f}.")
    return {
        "num": "RES 12", "order": 90.75, "name": "Crowding skeleton", "tech": "minimum spanning tree",
        "why": ("The merge tree drawn as geometry, decluttered: every entity is a dot, and only the "
                "bridges tighter than the bulk (median − 3 robust sd) are drawn — runs of them are the "
                "crowding skeleton, the paths along which entities blur into each other first. The roomy "
                "background needs no ink; it is the open space. Ringed points are the top pockets' "
                "members — same object as the pockets bars, seen spatially."),
        "svg": _svg(W, H, aria, head + '<g>' + dots + '</g><g>' + "".join(lines) + '</g><g>'
                    + "".join(rings) + "".join(labels) + '</g>' + foot),
        "legend": ('<span><i class="r"></i> tight bridge (tinted by tightness)</span>'
                   '<span><i class="f"></i> entity (no tight bridge)</span>'
                   '<span><i class="r"></i> pocket member (ringed)</span>'),
        "reveal": ("<b>Reveals:</b> <b>where</b> the tight structure sits — whether the pockets are one "
                   "region or scattered, and which bridge runs will blur first as neighborhoods tighten. "
                   "A healthy corpus draws few or no bridges at all."),
        "cls": "",
    }
