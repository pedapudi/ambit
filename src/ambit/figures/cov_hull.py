"""COV 07 — Reach boundary (convex hull).

The convex hull of the PCA-projected reservoir, drawn as a single accent polyline
over the faint cloud. The hull is the *outer reach* of the data in the plane; the
fraction of that reach actually occupied (vs. inflated, empty interior) is read off
a graphical split bar and the corner caption. Larger occupied fraction = the cloud
genuinely fills its envelope; a thin occupied wedge means the hull is bloated by a
few outliers and most of the enclosed area is void.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


def _hull_indices(P):
    """Return indices of the convex-hull vertices of P, in boundary order.

    scipy if available, else a monotone-chain (Andrew) fallback in numpy.
    """
    try:
        from scipy.spatial import ConvexHull
        return list(ConvexHull(P).vertices)
    except Exception:
        pts = np.asarray(P, float)
        order = np.lexsort((pts[:, 1], pts[:, 0]))

        def cross(o, a, b):
            return (pts[a, 0] - pts[o, 0]) * (pts[b, 1] - pts[o, 1]) \
                 - (pts[a, 1] - pts[o, 1]) * (pts[b, 0] - pts[o, 0])

        lower = []
        for i in order:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
                lower.pop()
            lower.append(int(i))
        upper = []
        for i in order[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
                upper.pop()
            upper.append(int(i))
        return lower[:-1] + upper[:-1]


def _shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@figure
def fig_cov_hull(ctx):
    W, H = 820, 560
    L, R, T, B = 96.0, 770.0, 70.0, 502.0  # plot frame (mirrors VIZ 07)

    xy = np.asarray(ctx.xy, float)
    # --- geometry in DATA space (for honest axis ticks) -------------------
    dmn, dmx = xy.min(0), xy.max(0)

    # --- geometry in SVG space (for drawing) ------------------------------
    P = _box(xy, W, H)  # (m,2) y already flipped, fit-to-width
    hull = _hull_indices(P)
    poly = P[hull]
    hull_area = _shoelace(poly)

    # occupied area: fraction of a fine grid (inside the plot box) whose cells
    # actually contain points -> occupied area, vs the convex hull area.
    gx, gy = 80, 56
    bx = np.clip(((P[:, 0] - L) / max(R - L, 1e-9) * gx).astype(int), 0, gx - 1)
    by = np.clip(((P[:, 1] - T) / max(B - T, 1e-9) * gy).astype(int), 0, gy - 1)
    occ_cells = len(set(zip(bx.tolist(), by.tolist())))
    cell_area = ((R - L) / gx) * ((B - T) / gy)
    occ_area = occ_cells * cell_area
    occ_frac = float(np.clip(occ_area / max(hull_area, 1e-9), 0.0, 1.0))
    inflated = 1.0 - occ_frac

    # density-aware faintness for the cloud (accumulation, no accent ramp)
    dens = _local_density(P, W, H)
    q = np.quantile(dens, 0.85) if len(dens) else 1.0

    body = ['<g font-family="ui-monospace, SFMono-Regular, Menlo, monospace">']

    # ---- R3: light Tufte reference axes + scale ticks (DATA units) -------
    nx_major, ny_major = 6, 4
    xticks = np.linspace(dmn[0], dmx[0], nx_major + 1)
    yticks = np.linspace(dmn[1], dmx[1], ny_major + 1)
    xpix = np.linspace(L, R, nx_major + 1)
    ypix = np.linspace(B, T, ny_major + 1)  # B=bottom (min), T=top (max)

    grid = ['<g stroke="var(--rule-soft)" stroke-width="1" '
            'vector-effect="non-scaling-stroke">']
    for xp in xpix:
        grid.append(f'<line x1="{xp:.1f}" y1="{B}" x2="{xp:.1f}" y2="{T}" opacity="0.5"/>')
    for yp in ypix:
        grid.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{R}" y2="{yp:.1f}" opacity="0.5"/>')
    grid.append('</g>')
    body.append("".join(grid))

    # solid left + bottom spines
    body.append('<g stroke="var(--rule)" stroke-width="1.2" '
                'vector-effect="non-scaling-stroke">'
                f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}"/>'
                f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}"/></g>')

    # minor ticks (4 minors between majors on x; 3 on y)
    minor = ['<g stroke="var(--rule-soft)" stroke-width="1" '
             'vector-effect="non-scaling-stroke">']
    for a, b in zip(xpix[:-1], xpix[1:]):
        for f in (0.25, 0.5, 0.75):
            xp = a + (b - a) * f
            minor.append(f'<line x1="{xp:.1f}" y1="{B}" x2="{xp:.1f}" y2="{B-4}"/>')
    for a, b in zip(ypix[:-1], ypix[1:]):
        for f in (0.33, 0.66):
            yp = a + (b - a) * f
            minor.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{L+4}" y2="{yp:.1f}"/>')
    minor.append('</g>')
    body.append("".join(minor))

    # major ticks + labels
    maj = ['<g stroke="var(--rule)" stroke-width="1.1" '
           'vector-effect="non-scaling-stroke">']
    for xp in xpix:
        maj.append(f'<line x1="{xp:.1f}" y1="{B}" x2="{xp:.1f}" y2="{B-7}"/>')
    for yp in ypix:
        maj.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{L+7}" y2="{yp:.1f}"/>')
    maj.append('</g>')
    body.append("".join(maj))

    labs = ['<g fill="var(--ink-faint)" font-size="9">']
    for xp, xv in zip(xpix, xticks):
        labs.append(f'<text x="{xp:.1f}" y="{B+14:.1f}" text-anchor="middle">{xv:+.2f}</text>')
    for yp, yv in zip(ypix, yticks):
        labs.append(f'<text x="{L-6:.1f}" y="{yp+3:.1f}" text-anchor="end">{yv:+.2f}</text>')
    labs.append('</g>')
    body.append("".join(labs))

    cx = (L + R) / 2
    cy = (T + B) / 2
    body.append(f'<text x="{cx:.1f}" y="{B+27:.1f}" fill="var(--ink-soft)" '
                'font-size="10" text-anchor="middle">PC-1 (projection units)</text>')
    body.append(f'<text x="{L-26:.1f}" y="{cy:.1f}" fill="var(--ink-soft)" font-size="10" '
                f'text-anchor="middle" transform="rotate(-90 {L-26:.1f} {cy:.1f})">'
                'PC-2 (projection units)</text>')

    # ---- faint cloud: neutral ink, density by accumulation --------------
    dots = ['<g fill="var(--ink-faint)">']
    for i in range(len(P)):
        op = 0.7 if dens[i] >= q else 0.42
        dots.append(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.2" '
                    f'fill-opacity="{op:.2f}"/>')
    dots.append('</g>')
    body.append("".join(dots))

    # ---- THE ONE accent: convex-hull reach boundary ---------------------
    hpath = " ".join(f"{poly[i,0]:.1f},{poly[i,1]:.1f}" for i in range(len(poly)))
    body.append(f'<polygon points="{hpath}" fill="none" stroke="var(--accent)" '
                'stroke-width="2.3" stroke-linejoin="round" '
                'vector-effect="non-scaling-stroke"/>')

    # hull vertices: the points that define the outer reach
    verts = ['<g fill="var(--ink)">']
    for i in range(len(poly)):
        verts.append(f'<circle cx="{poly[i,0]:.1f}" cy="{poly[i,1]:.1f}" r="2.4"/>')
    verts.append('</g>')
    body.append("".join(verts))

    # ---- R3: occupied-vs-inflated reach split bar (graphical) -----------
    # placed in the open top-left interior; widths reflect occ/inflated split.
    bar_x, bar_y, bar_w = L + 18, T + 14, 200.0
    occ_w = bar_w * occ_frac
    body.append('<g>'
                f'<text x="{bar_x:.1f}" y="{bar_y-6:.1f}" fill="var(--ink-soft)" '
                'font-size="9">reach budget · occupied vs convex hull</text>'
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{occ_w:.1f}" height="10" '
                'fill="var(--accent)" opacity="0.85"/>'
                f'<rect x="{bar_x+occ_w:.1f}" y="{bar_y:.1f}" width="{bar_w-occ_w:.1f}" '
                'height="10" fill="none" stroke="var(--ink-faint)" stroke-width="1" '
                'stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>'
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="10" '
                'fill="none" stroke="var(--rule)" stroke-width="1" '
                'vector-effect="non-scaling-stroke"/>'
                f'<line x1="{bar_x+occ_w:.1f}" y1="{bar_y-2:.1f}" x2="{bar_x+occ_w:.1f}" '
                f'y2="{bar_y+12:.1f}" stroke="var(--ink-soft)" stroke-width="1" '
                'vector-effect="non-scaling-stroke"/>'
                f'<text x="{bar_x+occ_w/2:.1f}" y="{bar_y+22:.1f}" fill="var(--accent)" '
                f'font-size="9" font-weight="700" text-anchor="middle">occupied {occ_frac*100:.0f}%</text>'
                f'<text x="{bar_x+occ_w+(bar_w-occ_w)/2:.1f}" y="{bar_y+22:.1f}" '
                f'fill="var(--ink-faint)" font-size="9" text-anchor="middle">inflated {inflated*100:.0f}%</text>'
                '</g>')

    # ---- title + corner captions ----------------------------------------
    body.append(f'<text x="{cx:.1f}" y="{T-12:.1f}" fill="var(--ink-soft)" '
                'font-size="11" text-anchor="middle">convex-hull reach boundary · '
                f'{len(poly)} vertices wrap the projected cloud</text>')
    body.append(f'<text x="{R+20:.1f}" y="{H-26:.1f}" fill="var(--accent)" font-size="12" '
                f'font-weight="700" text-anchor="end">occupied ≈ {occ_frac*100:.0f}% of convex hull</text>')
    body.append(f'<text x="{R+20:.1f}" y="{H-12:.1f}" fill="var(--ink-faint)" font-size="10" '
                f'text-anchor="end">inflated reach (hull − occupied) ≈ {inflated*100:.0f}% · '
                'enclosed area the data does not fill</text>')
    body.append(f'<text x="{L-26:.1f}" y="{H-12:.1f}" fill="var(--ink-faint)" font-size="10">'
                f'reservoir · PCA 2D · m={len(P):,}</text>')
    body.append('</g>')

    return {
        "num": "COV 07", "order": 7, "name": "Reach boundary (hull)", "tech": "convex hull",
        "why": (f"The convex hull wraps the projected reservoir with {len(poly)} vertices — its outer "
                f"reach in the plane. Only ≈{occ_frac*100:.0f}% of that enclosed area is actually "
                "occupied; the rest is inflated envelope from a few extremal points."),
        "svg": _svg(W, H, "Convex-hull reach boundary over the projected reservoir cloud, with an "
                          "occupied-vs-inflated split of the enclosed area", "".join(body)),
        "legend": ('<span><i class="f"></i> point (accumulates)</span>'
                   '<span><i class="a"></i> convex-hull boundary</span>'
                   '<span><i class="dash"></i> inflated (unoccupied) reach</span>'),
        "reveal": (f"<b>Reveals:</b> how much of the data's outer envelope is real coverage vs. empty "
                   f"space — here the cloud fills ≈{occ_frac*100:.0f}% of its convex hull, leaving "
                   f"≈{inflated*100:.0f}% as inflated reach stretched by extremal points."),
        "cls": "",
    }
