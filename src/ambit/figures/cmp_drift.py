"""CMP 13 — Drift field. The "*where* did the representation move" map: reuses the
CMP 10 grid scaffold (a 30×30 standardized-projection field with a colorbar and
projection-unit axes), but colors each cell by the **mean per-point drift** of the
A-points in it — ‖B − A‖ in A's projection frame — and overlays a sparse **quiver**
(one arrow per high-drift cell, from the cell's A-centroid to its B-centroid: the net
local movement). Needs two embeddings of equal dimension; reads only `ctx.cmp`.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg


def _placeholder(note):
    w, h = 760, 300
    body = (f'<text x="{w/2:.0f}" y="{h/2:.0f}" text-anchor="middle" font-size="12" '
            f'fill="var(--ink-faint)">{note}</text>')
    return {"num": "CMP 13", "order": 13, "name": "Drift field", "tech": "per-region movement",
            "why": "Where the representation moved between two embeddings of the same items.",
            "svg": _svg(w, h, f"Drift field unavailable: {note}.", body),
            "legend": f'<span><i class="f"></i> {note}</span>',
            "reveal": f"<b>Reveals:</b> {note}.", "cls": ""}


@figure
def fig_cmp_drift(ctx):
    cmp = getattr(ctx, "cmp", None)
    if cmp is None:
        return _placeholder("needs a second embedding — run with --compare")
    if not cmp.same_dim or cmp.xyB_in_A is None:
        return _placeholder("drift field needs equal dims (d_A ≠ d_B) — see the CKA scorecard")

    w, h = 760, 690
    x0, y0 = 60.0, 60.0
    fw = 540.0
    G = 30
    cw = fw / G

    # standardize A's projection frame; map B through the SAME frame so drift is comparable
    xyA = np.asarray(cmp.xyA, float)
    xyB = np.asarray(cmp.xyB_in_A, float)
    mu = xyA.mean(0)
    sd = xyA.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    ZA = (xyA - mu) / sd
    ZB = (xyB - mu) / sd
    drift = np.linalg.norm(ZB - ZA, axis=1)                  # standardized drift per point
    R = 3.0

    gx = np.floor((ZA[:, 0] + R) / (2 * R) * G).astype(int)
    gy = np.floor((ZA[:, 1] + R) / (2 * R) * G).astype(int)
    inside = (gx >= 0) & (gx < G) & (gy >= 0) & (gy < G)

    gi, gj = gx[inside], gy[inside]
    cnt = np.zeros((G, G)); np.add.at(cnt, (gi, gj), 1.0)
    acx = np.zeros((G, G)); np.add.at(acx, (gi, gj), ZA[inside, 0])   # A centroid accum
    acy = np.zeros((G, G)); np.add.at(acy, (gi, gj), ZA[inside, 1])
    bcx = np.zeros((G, G)); np.add.at(bcx, (gi, gj), ZB[inside, 0])   # B centroid accum
    bcy = np.zeros((G, G)); np.add.at(bcy, (gi, gj), ZB[inside, 1])
    sumd = np.zeros((G, G)); np.add.at(sumd, (gi, gj), drift[inside])
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_d = np.where(cnt > 0, sumd / np.maximum(cnt, 1), np.nan)

    active = cnt > 0
    dvals = cell_d[active]
    dhi = float(np.quantile(dvals, 0.95)) if dvals.size else 1.0
    dhi = dhi or 1.0
    thr = float(np.quantile(dvals, 0.60)) if dvals.size else 0.0

    def cell_px(i, j):
        return x0 + i * cw, y0 + (G - 1 - j) * cw

    def u_to_px(ux, uy):
        return x0 + (ux + R) / (2 * R) * fw, y0 + (R - uy) / (2 * R) * fw

    body = []
    body.append(f'<text x="{x0:.0f}" y="34" fill="var(--ink-soft)" font-size="12">'
                f'drift field · mean ‖{cmp.label_b} − A‖ per cell, in A\'s projection frame</text>')
    body.append(f'<text x="{x0:.0f}" y="48" fill="var(--ink-faint)" font-size="10">'
                f'{G}×{G} standardized cells · arrows show net local movement (A-centroid → '
                f'{cmp.label_b}-centroid) where drift is high</text>')

    # cell fills: sequential ink-faint -> accent by mean drift
    for i in range(G):
        for j in range(G):
            px, py = cell_px(i, j)
            if not active[i, j]:
                continue
            mag = float(np.clip(cell_d[i, j] / dhi, 0.0, 1.0))
            pct = int(round(8 + mag * 92))
            body.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{cw:.2f}" height="{cw:.2f}" '
                        f'fill="color-mix(in srgb, var(--accent) {pct}%, transparent)" shape-rendering="crispEdges"/>')

    body.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{fw:.2f}" height="{fw:.2f}" fill="none" '
                f'stroke="var(--rule)" stroke-width="1" vector-effect="non-scaling-stroke"/>')

    # quiver: one arrow per high-drift cell, A-centroid -> B-centroid (scaled for legibility)
    AMP = 1.0
    for i in range(G):
        for j in range(G):
            if not active[i, j] or cnt[i, j] < 3 or cell_d[i, j] < thr:
                continue
            ax, ay = acx[i, j] / cnt[i, j], acy[i, j] / cnt[i, j]
            bx, by = bcx[i, j] / cnt[i, j], bcy[i, j] / cnt[i, j]
            px0, py0 = u_to_px(ax, ay)
            px1, py1 = u_to_px(ax + (bx - ax) * AMP, ay + (by - ay) * AMP)
            dx, dy = px1 - px0, py1 - py0
            ln = (dx * dx + dy * dy) ** 0.5 or 1.0
            if ln < 2.0:
                continue
            ux, uy = dx / ln, dy / ln
            hx, hy = px1 - 5 * ux, py1 - 5 * uy
            nx, ny = -uy, ux
            body.append(f'<line x1="{px0:.1f}" y1="{py0:.1f}" x2="{px1:.1f}" y2="{py1:.1f}" '
                        f'stroke="var(--ink)" stroke-width="1" opacity="0.75" vector-effect="non-scaling-stroke"/>')
            body.append(f'<path d="M {px1:.1f} {py1:.1f} L {hx+3*nx:.1f} {hy+3*ny:.1f} '
                        f'L {hx-3*nx:.1f} {hy-3*ny:.1f} Z" fill="var(--ink)" opacity="0.85"/>')

    # axes ticks
    ax_b = y0 + fw
    for u in range(-3, 4):
        xx, _ = u_to_px(u, 0)
        _, yy = u_to_px(0, u)
        body.append(f'<text x="{xx:.1f}" y="{ax_b+15:.1f}" fill="var(--ink-faint)" font-size="9" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{u:+d}</text>'.replace('+0', '0'))
        body.append(f'<text x="{x0-8:.1f}" y="{yy+3:.1f}" fill="var(--ink-faint)" font-size="9" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{u:+d}</text>'.replace('+0', '0'))
    body.append(f'<text x="{x0+fw/2:.1f}" y="{ax_b+30:.1f}" fill="var(--ink-soft)" font-size="9.5" '
                f'text-anchor="middle">projection x · standardized units</text>')

    # colorbar (drift magnitude)
    bx, bw, bh = 626.0, 14.0, fw
    steps = 60
    for s in range(steps):
        mag = 1.0 - (s + 0.5) / steps
        pct = int(round(8 + mag * 92))
        body.append(f'<rect x="{bx:.1f}" y="{y0 + s*bh/steps:.2f}" width="{bw:.0f}" '
                    f'height="{bh/steps + 0.6:.2f}" fill="color-mix(in srgb, var(--accent) {pct}%, transparent)" '
                    f'shape-rendering="crispEdges"/>')
    body.append(f'<rect x="{bx:.1f}" y="{y0:.1f}" width="{bw:.0f}" height="{bh:.1f}" fill="none" '
                f'stroke="var(--rule-soft)" stroke-width="0.5"/>')
    body.append(f'<text x="{bx+bw+5:.1f}" y="{y0+6:.1f}" fill="var(--ink-faint)" font-size="9" '
                f'style="font-variant-numeric:tabular-nums">{dhi:.2f}</text>')
    body.append(f'<text x="{bx+bw+5:.1f}" y="{y0+bh:.1f}" fill="var(--ink-faint)" font-size="9">0</text>')
    body.append(f'<text x="{bx+bw/2:.1f}" y="{y0-8:.1f}" fill="var(--ink-soft)" font-size="8" '
                f'text-anchor="middle">drift</text>')

    body.append(f'<text x="{x0:.0f}" y="{ax_b+52:.0f}" fill="var(--ink-faint)" font-size="9.5">'
                f'deeper = that region of A moved further to become {cmp.label_b}; arrows give the '
                f'net direction. No current single-set figure localizes change like this.</text>')

    mean_drift = float(drift.mean())
    aria = (f"Drift field: a 30×30 standardized grid over embedding A's projection, each cell shaded by "
            f"the mean distance its points moved to become {cmp.label_b} (mean drift {mean_drift:.2f}), "
            f"with quiver arrows from each high-drift cell's A-centroid to its {cmp.label_b}-centroid "
            f"showing the net local direction of movement.")
    return {
        "num": "CMP 13", "order": 13, "name": "Drift field", "tech": "per-region movement",
        "why": f"Where the representation moved from A to {cmp.label_b}: each cell is shaded by how far "
               f"its points drifted (mean {mean_drift:.2f} in standardized projection units), and the "
               f"arrows show the net local direction. This localizes *change* — the largest arrows mark "
               f"the regions the second embedding reshaped, which no single-set figure can show.",
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": '<span><i class="a"></i> mean per-region drift (deeper = moved further)</span>'
                  '<span><i class="dash"></i> quiver — net local movement A → ' + cmp.label_b + '</span>',
        "reveal": "<b>Reveals:</b> <b>where</b> the representation moved — a knot of deep cells and long "
                  "arrows is a region the second embedding reshaped; a flat, pale field means the change was "
                  "uniform. Pairs with the CKA scorecard (how much) to answer where.",
        "cls": "",
    }
