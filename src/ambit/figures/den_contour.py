"""DEN 02 — Isodensity contour relief.

ctx.xy is binned into a grid and smoothed into a continuous density field; nested
closed isodensity level-sets (marching squares) are drawn as stacked bands. This is
a legitimate density heatmap, so rising density mixes the accent->bad ramp; the
interior low-density void reads as bare paper ground with a dashed zero-density edge.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _gf  # type: ignore
except Exception:  # pragma: no cover - scipy optional
    _gf = None


def _smooth(field, sigma):
    """Gaussian blur of a 2-D field; pure-numpy separable fallback if no scipy."""
    if _gf is not None:
        return _gf(field, sigma=sigma, mode="constant")
    rad = max(1, int(3 * sigma))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    out = field.astype(float)
    # blur rows then columns
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
    return out


def _segments(grid, level):
    """Marching-squares line segments for one iso `level` of a (ny,nx) `grid`.

    Returns a list of ((x0,y0),(x1,y1)) in *grid* coordinates (col,row floats).
    """
    ny, nx = grid.shape
    segs = []

    def interp(va, vb, a, b):
        d = vb - va
        t = 0.5 if abs(d) < 1e-12 else (level - va) / d
        t = min(1.0, max(0.0, t))
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    for j in range(ny - 1):
        for i in range(nx - 1):
            tl = grid[j, i]; tr = grid[j, i + 1]
            bl = grid[j + 1, i]; br = grid[j + 1, i + 1]
            idx = (tl > level) | ((tr > level) << 1) | ((br > level) << 2) | ((bl > level) << 3)
            idx = int(idx)
            if idx == 0 or idx == 15:
                continue
            ptl = (i, j); ptr = (i + 1, j); pbl = (i, j + 1); pbr = (i + 1, j + 1)
            # edge midpoints by interpolation
            top = interp(tl, tr, ptl, ptr)
            right = interp(tr, br, ptr, pbr)
            bottom = interp(bl, br, pbl, pbr)
            left = interp(tl, bl, ptl, pbl)
            e = {
                1: [(left, top)], 2: [(top, right)], 3: [(left, right)],
                4: [(right, bottom)], 5: [(left, bottom), (top, right)],
                6: [(top, bottom)], 7: [(left, bottom)], 8: [(bottom, left)],
                9: [(bottom, top)], 10: [(top, left), (bottom, right)],
                11: [(bottom, right)], 12: [(right, left)], 13: [(right, top)],
                14: [(top, left)],
            }.get(idx, [])
            segs.extend(e)
    return segs


def _polylines(segs, tol=1e-6):
    """Stitch unordered segments into closed/open polylines for smooth filled paths."""
    if not segs:
        return []
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
    adj = {}
    for a, b in segs:
        adj.setdefault(key(a), []).append((a, b))
        adj.setdefault(key(b), []).append((b, a))
    used = [False] * len(segs)
    seg_at = {}
    for n, (a, b) in enumerate(segs):
        seg_at.setdefault(key(a), []).append((n, a, b))
        seg_at.setdefault(key(b), []).append((n, b, a))
    chains = []
    for start in range(len(segs)):
        if used[start]:
            continue
        a, b = segs[start]
        used[start] = True
        chain = [a, b]
        # extend forward from b
        cur = b
        while True:
            nxt = None
            for n, p, q in seg_at.get(key(cur), []):
                if not used[n]:
                    nxt = (n, q); break
            if nxt is None:
                break
            used[nxt[0]] = True
            chain.append(nxt[1]); cur = nxt[1]
            if key(cur) == key(chain[0]):
                break
        chains.append(chain)
    return chains


@figure
def fig_den_contour(ctx):
    w, h = 760, 470
    pad = 22
    P = _box(ctx.xy, w, h, pad)

    # --- density field on a grid over the svg box --------------------------------
    gx, gy = 96, 60  # field columns x rows (fine for smooth contours)
    x0, x1 = pad, w - pad
    y0, y1 = pad, h - pad
    bx = np.clip(((P[:, 0] - x0) / (x1 - x0) * (gx - 1)).astype(int), 0, gx - 1)
    by = np.clip(((P[:, 1] - y0) / (y1 - y0) * (gy - 1)).astype(int), 0, gy - 1)
    grid = np.zeros((gy, gx))
    np.add.at(grid, (by, bx), 1.0)
    field = _smooth(grid, sigma=2.0)
    if field.max() <= 0:
        field[:] = 1e-9
    fmax = float(field.max())

    # grid->svg mappers
    def gx2x(c):
        return x0 + c / (gx - 1) * (x1 - x0)

    def gy2y(r):
        return y0 + r / (gy - 1) * (y1 - y0)

    # --- nested isodensity levels as density quantiles of the field --------------
    occ = field[field > fmax * 0.02]
    if occ.size < 8:
        occ = field.ravel()
    qs = [0.30, 0.50, 0.65, 0.78, 0.88, 0.95]
    levels = [float(np.quantile(occ, q)) for q in qs]
    # de-duplicate / enforce strictly increasing
    lv = []
    for v in levels:
        if not lv or v > lv[-1] * 1.0001:
            lv.append(v)
    levels = lv
    nb = len(levels)

    body = []

    # accent->bad ramp: empty=ground (paper), rising density carries more color.
    # band k (outer..inner) fills with increasing color-mix(bad,accent) + opacity.
    def band_fill(k):
        # k from 0 (faint, low density) to nb-1 (peak)
        frac = k / max(1, nb - 1)
        badpct = int(round(8 + frac * 78))  # 8%..86% bad mixed into accent
        return f"color-mix(in srgb, var(--bad) {badpct}%, var(--accent))"

    def band_opacity(k):
        return 0.12 + 0.072 * k

    # outermost contour = effective support boundary (dashed zero-density edge)
    base_segs = _segments(field, levels[0])
    base_chains = _polylines(base_segs)

    # --- stacked filled bands, outer (low) first so inner (high) paints on top ---
    for k, level in enumerate(levels):
        segs = _segments(field, level)
        chains = _polylines(segs)
        if not chains:
            continue
        d = []
        for ch in chains:
            if len(ch) < 3:
                continue
            pts = [(gx2x(c), gy2y(r)) for (c, r) in ch]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            if (max(xs) - min(xs)) < 6 and (max(ys) - min(ys)) < 6:
                continue  # drop sub-6px specks, keep legitimate small rings
            d.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z")
        if not d:
            continue
        path = " ".join(d)
        body.append(
            f'<path d="{path}" fill="{band_fill(k)}" fill-opacity="{band_opacity(k):.3f}" '
            f'fill-rule="evenodd" stroke="{band_fill(k)}" stroke-opacity="{min(0.9,0.3+0.1*k):.2f}" '
            f'stroke-width="{0.6 + 0.12*k:.2f}" vector-effect="non-scaling-stroke"/>'
        )

    # --- dashed zero-density / effective-support edge (drawn last, on top) -------
    if base_chains:
        de = []
        for ch in base_chains:
            if len(ch) < 3:
                continue
            pts = [(gx2x(c), gy2y(r)) for (c, r) in ch]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            if (max(xs) - min(xs)) < 6 and (max(ys) - min(ys)) < 6:
                continue
            de.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z")
        if de:
            body.append(
                f'<path d="{" ".join(de)}" fill="none" stroke="var(--ink-soft)" '
                f'stroke-width="1" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>'
            )

    # --- peak marker: the single accent emphasis at the global density maximum ---
    pj, pi = np.unravel_index(int(np.argmax(field)), field.shape)
    px, py = gx2x(pi), gy2y(pj)
    peak_ratio = fmax / max(1e-9, float(np.median(occ)))
    body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.4" fill="var(--accent)"/>')
    body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="none" '
                f'stroke="var(--accent)" stroke-width="1" stroke-opacity="0.5"/>')
    lbl_anchor = "end" if px > x1 - 90 else "start"
    lbl_x = px - 9 if lbl_anchor == "end" else px + 9
    body.append(f'<text x="{lbl_x:.1f}" y="{py-7:.1f}" font-size="9.5" fill="var(--accent)" '
                f'text-anchor="{lbl_anchor}" '
                f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">density peak</text>')

    # --- fine axes: evenly spaced labeled ticks on the projected coordinates -----
    xmn, ymn = ctx.xy.min(0)
    xmx, ymx = ctx.xy.max(0)
    body.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="var(--rule)" stroke-width="1"/>')
    body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="var(--rule)" stroke-width="1"/>')
    for t in np.linspace(0, 1, 7):
        xx = x0 + t * (x1 - x0)
        val = xmn + t * (xmx - xmn)
        body.append(f'<line x1="{xx:.1f}" y1="{y1}" x2="{xx:.1f}" y2="{y1+4}" stroke="var(--rule-soft)"/>')
        body.append(f'<text x="{xx:.1f}" y="{y1+15:.1f}" font-size="9" fill="var(--ink-faint)" '
                    f'text-anchor="middle" font-family="ui-monospace, monospace">{val:+.2f}</text>')
    for t in np.linspace(0, 1, 5):
        yy = y1 - t * (y1 - y0)
        val = ymn + t * (ymx - ymn)
        body.append(f'<line x1="{x0-4}" y1="{yy:.1f}" x2="{x0}" y2="{yy:.1f}" stroke="var(--rule-soft)"/>')
        body.append(f'<text x="{x0-6:.1f}" y="{yy+3:.1f}" font-size="9" fill="var(--ink-faint)" '
                    f'text-anchor="end" font-family="ui-monospace, monospace">{val:+.2f}</text>')
    body.append(f'<text x="{(x0+x1)/2:.1f}" y="{h-3}" font-size="9" fill="var(--ink-faint)" '
                f'text-anchor="middle">PC1</text>')
    body.append(f'<text x="11" y="{(y0+y1)/2:.1f}" font-size="9" fill="var(--ink-faint)" '
                f'text-anchor="middle" transform="rotate(-90 11 {(y0+y1)/2:.1f})">PC2</text>')

    # --- density-quantile scale: stepped swatches of the accent->bad ramp --------
    sx, sy = 130, 34
    body.append('<text x="22" y="40" font-size="8.5" fill="var(--ink-faint)" '
                'text-anchor="start" font-family="ui-monospace, monospace">density quantile</text>')
    body.append('<g font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="8" '
                'font-variant-numeric="tabular-nums" text-anchor="middle">')
    qlabs = [int(round(q * 100)) for q in qs[:nb]]
    for k in range(nb):
        rx = sx + k * 24
        body.append(f'<rect x="{rx}" y="{sy}" width="22" height="8" fill="{band_fill(k)}" '
                    f'fill-opacity="{band_opacity(k)+0.05:.3f}" stroke="var(--ink-faint)" stroke-width="0.6"/>')
        col = "var(--accent)" if k == nb - 1 else "var(--ink-faint)"
        body.append(f'<text x="{rx+11}" y="{sy+16}" fill="{col}">{qlabs[k]}</text>')
    body.append(f'<text x="{sx + nb*24 + 4}" y="{sy+16}" fill="var(--ink-faint)" '
                f'text-anchor="start">% — inner band (accent) = densest peak</text>')
    body.append('</g>')

    aria = ("Isodensity contour relief of the projected embedding cloud: nested closed "
            "level-sets at fixed density quantiles. Stacked bands rise from a faint outer "
            "support boundary to a saturated peak ring at the global density maximum (the one "
            "accent marker); the low-density interior reads as bare paper ground, edged by a "
            "dashed zero-density contour.")

    erank = metrics.effective_rank(ctx.eigs)
    return {
        "num": "DEN 02", "order": 2, "name": "Isodensity contour relief", "tech": "kde · contours",
        "why": (f"The reservoir's projected density binned and smoothed into a field, then sliced "
                f"into nested isodensity level-sets. The bare interior is genuinely empty ground; "
                f"tight inner rings mark a concentrated peak ≈{peak_ratio:.0f}× the median occupied cell."),
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": '<span><i class="a"></i> peak ring / density max (accent)</span>'
                  '<span><i class="dash"></i> zero-density edge (effective support)</span>'
                  '<span><i class="f"></i> paper interior = empty ground</span>',
        "reveal": (f"<b>Reveals:</b> the shape of occupied space — where mass concentrates into "
                   f"peaks versus the void it leaves empty, across ≈{erank:.0f} effective dimensions "
                   f"collapsed to 2-D."),
        "cls": "",
    }
