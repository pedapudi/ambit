from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_den_hexbin(ctx):
    """DEN 03 — Hexbin occupancy.

    Aggregate the PCA-projected reservoir into pointy-top hexagonal bins, tint
    each occupied cell by its point count via an accent->bad color-mix ramp
    (sparse->dense). Empty hexes are not drawn. A discrete count-ramp legend with
    numbered breakpoints sits top-right; the peak cell carries an accent outline.
    """
    W, H = 920, 560
    # plot box (leave room for axes left/bottom, ramp legend right, captions top)
    L, R, T, B = 60.0, 820.0, 44.0, 482.0  # inner data rect
    bw, bh = R - L, B - T

    xy = np.asarray(ctx.xy, float)
    if xy is None or len(xy) == 0:
        note = '<text x="%.0f" y="%.0f" fill="var(--ink-faint)" font-size="12" text-anchor="middle">needs projected reservoir</text>' % (W / 2, H / 2)
        return {"num": "DEN 03", "order": 3, "name": "Hexbin occupancy", "tech": "hexbin",
                "why": "No projected reservoir available to bin.",
                "svg": _svg(W, H, "Hexbin occupancy (no data)", note),
                "legend": "", "reveal": "<b>Reveals:</b> per-cell occupancy of the projected space.", "cls": ""}

    # ---- map data coords into the inner rect (y flipped), preserving aspect ----
    mn = xy.min(0)
    span = np.maximum(xy.max(0) - mn, 1e-9)
    # uniform scale so hexes stay regular; fit the larger extent to its axis
    sx = bw / span[0]
    sy = bh / span[1]
    s = min(sx, sy)
    ox = L + (bw - s * span[0]) / 2.0
    oy = T + (bh - s * span[1]) / 2.0
    Px = ox + (xy[:, 0] - mn[0]) * s
    # y flipped within the centered band: data-min at bottom (B), data-max at top
    Py = (T + bh) - ((xy[:, 1] - mn[1]) * s) - (bh - s * span[1]) / 2.0

    # ---- pointy-top hex lattice ----
    # choose a column count, derive radius; pointy-top: width=sqrt(3)*r, vstep=1.5*r
    ncol = 26
    hw = bw / ncol                      # horizontal spacing between hex centers
    r = hw / np.sqrt(3.0)               # circumradius for pointy-top
    vstep = 1.5 * r
    nrow = int(np.ceil(bh / vstep)) + 1

    # assign each point to nearest hex center on the offset lattice.
    # offset-coords: row index, col index (cols shifted half-width on odd rows)
    # approximate axial binning then snap.
    # center of cell (row j, col i): cx = L + hw*(i+0.5) + (hw/2 if j odd), cy = T + vstep*(j+0.5)
    j = np.clip(((Py - T) / vstep - 0.5).round().astype(int), 0, nrow - 1)
    odd = (j & 1).astype(float)
    i = np.clip(((Px - L) / hw - 0.5 - 0.5 * odd).round().astype(int), 0, ncol - 1)
    key = j * (ncol + 2) + i

    uk, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    # recover (i,j) for each unique cell
    cj = uk // (ncol + 2)
    ci = uk % (ncol + 2)
    cx = L + hw * (ci + 0.5) + (hw / 2.0) * (cj & 1)
    cy = T + vstep * (cj + 0.5)

    cmax = int(counts.max())
    cmin = int(counts.min())
    peak = int(np.argmax(counts))

    # ---- pointy-top hex polygon vertices (apex up/down) ----
    # for pointy-top, vertices at angles 30,90,...; width sqrt3*r, height 2r
    def hexpts(x, y):
        ax = np.sqrt(3.0) / 2.0 * r
        return (f"{x:.1f},{y - r:.1f} {x + ax:.1f},{y - r / 2:.1f} "
                f"{x + ax:.1f},{y + r / 2:.1f} {x:.1f},{y + r:.1f} "
                f"{x - ax:.1f},{y - r / 2:.1f} {x - ax:.1f},{y + r / 2:.1f}")

    # tint ramp: sparse (few) -> dense (many) == accent -> bad.
    # log-scaled so the long sparse tail stays distinguishable from voids.
    lo = np.log1p(cmin)
    hi = np.log1p(cmax)
    rng = max(hi - lo, 1e-9)

    def tint(c):
        t = (np.log1p(c) - lo) / rng           # 0..1, dense->1
        pct = int(round(8 + 92 * t))           # keep a floor so sparse cells read
        return f"color-mix(in srgb, var(--bad) {pct}%, var(--accent))"

    hexes = []
    for u in range(len(uk)):
        x, y, c = float(cx[u]), float(cy[u]), int(counts[u])
        hexes.append(
            f'<polygon points="{hexpts(x, y)}" fill="{tint(c)}" '
            f'stroke="var(--rule-soft)" stroke-width="0.5" vector-effect="non-scaling-stroke"/>')

    # accent outline on the single peak cell
    px, py = float(cx[peak]), float(cy[peak])
    peak_ring = (f'<polygon points="{hexpts(px, py)}" fill="none" stroke="var(--accent)" '
                 f'stroke-width="2.2" vector-effect="non-scaling-stroke"/>')
    peak_lbl = (f'<text x="{px:.1f}" y="{py + 3.5:.1f}" fill="var(--paper)" font-size="10" '
                f'font-weight="700" text-anchor="middle" '
                f'style="font-variant-numeric:tabular-nums">{cmax}</text>')

    # ---- fine axes: ticks + mono labels along left (proj-y) and bottom (proj-x) ----
    ax = []
    ax.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule-soft)" stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
    ax.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="var(--rule-soft)" stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
    nticks = 6
    x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
    y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
    for t in range(nticks + 1):
        f = t / nticks
        # x axis (data x left->right)
        xx = ox + f * (s * span[0])
        vx = x0 + f * (x1 - x0)
        ax.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B + 6}" stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
        ax.append(f'<text x="{xx:.1f}" y="{B + 17}" fill="var(--ink-faint)" font-size="9" text-anchor="middle" style="font-variant-numeric:tabular-nums">{vx:+.2f}</text>')
        # y axis (data y bottom->top); top of centered band = oy
        yy = (T + bh) - (bh - s * span[1]) / 2.0 - f * (s * span[1])
        vy = y0 + f * (y1 - y0)
        ax.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L - 6}" y2="{yy:.1f}" stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
        ax.append(f'<text x="{L - 9}" y="{yy + 3.2:.1f}" fill="var(--ink-faint)" font-size="9" text-anchor="end" style="font-variant-numeric:tabular-nums">{vy:+.2f}</text>')
    ax.append(f'<text x="{(L + R) / 2:.1f}" y="{B + 32}" fill="var(--ink-soft)" font-size="9" text-anchor="middle">projection · PC-1 (column axis)</text>')
    ax.append(f'<text x="28" y="{(T + B) / 2:.1f}" fill="var(--ink-soft)" font-size="9" text-anchor="middle" transform="rotate(-90 28 {(T + B) / 2:.1f})">projection · PC-2 (row axis)</text>')

    # ---- header captions ----
    occ = len(uk)
    total = int(counts.sum())
    head = (
        f'<text x="{L + 10}" y="34" fill="var(--ink-soft)" font-size="11">'
        f'hexbin occupancy · per-cell point count · sparse→dense = accent→bad</text>'
        f'<text x="{R}" y="34" fill="var(--accent)" font-size="11" font-weight="700" '
        f'text-anchor="end">peak cell n={cmax} · {occ} occupied / {ncol}×{nrow} lattice</text>')

    # ---- discrete count-ramp legend (top-right column), numbered breakpoints ----
    # breakpoints span min..max on the same log ramp the cells use.
    lx, ly0 = 858.0, 73.5
    sw = 8.0   # swatch half-width region
    leg = ['<text x="%.1f" y="58.0" fill="var(--ink-soft)" font-size="9">count n</text>' % (lx + 12)]
    leg.append('<text x="%.1f" y="69.0" fill="var(--ink-faint)" font-size="7.5">per cell · ◇=peak</text>' % (lx + 12))

    # numbered breakpoints: peak, then descending round-ish steps, down to min, plus a void '0'.
    bps = sorted({cmax, max(int(cmax * 0.66), 1), max(int(cmax * 0.4), 1),
                  max(int(cmax * 0.2), 1), max(int(cmax * 0.08), 1), cmin}, reverse=True)
    aL = np.sqrt(3.0) / 2.0 * 7.0
    y = ly0
    for bi, c in enumerate(bps):
        cxs, cys = lx + 6, y
        pts = (f"{cxs:.1f},{cys - 7:.1f} {cxs + aL:.1f},{cys - 3.5:.1f} "
               f"{cxs + aL:.1f},{cys + 3.5:.1f} {cxs:.1f},{cys + 7:.1f} "
               f"{cxs - aL:.1f},{cys - 3.5:.1f} {cxs - aL:.1f},{cys + 3.5:.1f}")
        leg.append(f'<polygon points="{pts}" fill="{tint(c)}" stroke="var(--rule-soft)" '
                   f'stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
        if c == cmax:
            ring = (f"{cxs:.1f},{cys - 8.2:.1f} {cxs + aL + 1.3:.1f},{cys - 4.1:.1f} "
                    f"{cxs + aL + 1.3:.1f},{cys + 4.1:.1f} {cxs:.1f},{cys + 8.2:.1f} "
                    f"{cxs - aL - 1.3:.1f},{cys - 4.1:.1f} {cxs - aL - 1.3:.1f},{cys + 4.1:.1f}")
            leg.append(f'<polygon points="{ring}" fill="none" stroke="var(--accent)" '
                       f'stroke-width="1.6" vector-effect="non-scaling-stroke"/>')
        leg.append(f'<text x="{cxs + 16:.1f}" y="{cys + 3.5:.1f}" fill="var(--ink)" font-size="8.5" '
                   f'style="font-variant-numeric:tabular-nums">{c}</text>')
        y += 19.5
    # void swatch (bare paper, dashed) — empty hexes are not drawn in-plot
    cys = y
    pts = (f"{lx + 6:.1f},{cys - 7:.1f} {lx + 6 + aL:.1f},{cys - 3.5:.1f} "
           f"{lx + 6 + aL:.1f},{cys + 3.5:.1f} {lx + 6:.1f},{cys + 7:.1f} "
           f"{lx + 6 - aL:.1f},{cys - 3.5:.1f} {lx + 6 - aL:.1f},{cys + 3.5:.1f}")
    leg.append(f'<polygon points="{pts}" fill="var(--paper)" stroke="var(--ink-faint)" '
               f'stroke-width="0.6" stroke-dasharray="2 2" vector-effect="non-scaling-stroke"/>')
    leg.append(f'<text x="{lx + 22:.1f}" y="{cys + 3.5:.1f}" fill="var(--ink-faint)" font-size="8.5" '
               f'style="font-variant-numeric:tabular-nums">0</text>')

    body = (
        head
        + '<g aria-hidden="true" font-family="inherit">' + "".join(ax) + '</g>'
        + '<g>' + "".join(hexes) + '</g>'
        + '<g>' + peak_ring + '</g>'
        + '<g>' + peak_lbl + '</g>'
        + '<g aria-hidden="true" font-family="inherit">' + "".join(leg) + '</g>')

    aria = (f"Hexbin occupancy of the projected reservoir: a honeycomb of {occ} occupied "
            f"pointy-top cells over {total} points, each tinted by its point count "
            f"(sparse to dense, accent to bad); the peak cell n={cmax} carries an accent "
            f"outline; empty cells are left as bare paper.")

    return {
        "num": "DEN 03", "order": 3, "name": "Hexbin occupancy", "tech": "hexbin",
        "why": ("The projected reservoir aggregated into hexagonal bins; each occupied cell tints by "
                "its point count so concentration reads as heat and empty space stays paper-bare."),
        "svg": _svg(W, H, aria, body),
        "legend": ('<span><i class="a"></i> dense cell (accent→bad ramp)</span>'
                   '<span><i class="f"></i> sparse cell</span>'
                   '<span><i class="r"></i> empty / void (undrawn)</span>'),
        "reveal": ("<b>Reveals:</b> how unevenly the corpus packs the projected space — where it "
                   "piles into a few hot cells versus the many sparse and empty bins around them."),
        "cls": "",
    }
