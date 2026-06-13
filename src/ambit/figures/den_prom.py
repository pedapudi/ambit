from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


def _blur(field):
    """One separable 3-tap box pass on a 2-D field (pure numpy, edge-clamped)."""
    out = field
    for ax in (0, 1):
        out = (np.roll(out, 1, ax) + out + np.roll(out, -1, ax)) / 3.0
    return out


def _march_segments(field, level):
    """Marching-squares line segments for one iso `level` of a (gx,gy) `field`.

    Returns ((x0,y0),(x1,y1)) pairs in *grid* coordinates (col i, row j as floats).
    Standard 16-case table; corners that exceed `level` set their bit, segment
    endpoints are linearly interpolated along the crossed cell edges.
    """
    gx, gy = field.shape
    segs = []

    def interp(va, vb, a, b):
        d = vb - va
        t = 0.5 if abs(d) < 1e-12 else (level - va) / d
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    for i in range(gx - 1):
        for j in range(gy - 1):
            tl = field[i, j]; tr = field[i + 1, j]
            bl = field[i, j + 1]; br = field[i + 1, j + 1]
            idx = ((tl > level)) | ((tr > level) << 1) | ((br > level) << 2) | ((bl > level) << 3)
            idx = int(idx)
            if idx == 0 or idx == 15:
                continue
            ptl = (i, j); ptr = (i + 1, j); pbl = (i, j + 1); pbr = (i + 1, j + 1)
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


def _stitch(segs, tol=1e-6):
    """Stitch unordered marching-squares segments into ordered polylines."""
    if not segs:
        return []
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
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
def fig_den_prom(ctx):
    """Density-peak prominence over a topographic relief: the projected density field
    is drawn as colour-graded isodensity contour lines (marching squares), and grid
    local maxima are ranked by topographic prominence. Each peak is a dot over the
    faint cloud sized by its prominence; the single dominant peak carries the accent.
    A fine prominence axis on the right ranks every peak against a noise cutoff."""
    W, H = 900, 488
    # left panel = relief + sited peaks ; right panel = prominence ranking
    Lx0, Lx1 = 30, 530          # cloud box
    Rx0, Rx1 = 606, 818         # ranking bar field
    pad = 24

    # ---- project the reservoir into the cloud box -------------------------
    P = _box(ctx.xy, Lx1 - Lx0, H, pad=pad)
    P[:, 0] += Lx0              # shift into left panel

    # ---- density field on a grid (2-D histogram + a couple of blur passes) -
    gx, gy = 84, 60
    cw, ch = (Lx1 - Lx0), H
    bx = np.clip(((P[:, 0] - Lx0) / cw * gx).astype(int), 0, gx - 1)
    by = np.clip((P[:, 1] / ch * gy).astype(int), 0, gy - 1)
    grid = np.zeros((gx, gy))
    np.add.at(grid, (bx, by), 1.0)
    sm = _blur(_blur(_blur(grid)))          # three light box passes -> smooth field
    occ = sm > 0
    mean_occ = sm[occ].mean() if occ.any() else 1.0
    fmax = float(sm.max()) if sm.max() > 0 else 1.0

    # grid cell -> svg coords
    def cell_xy(i, j):
        return (Lx0 + (i + 0.5) / gx * cw, (j + 0.5) / gy * ch)

    def gi2x(c):
        return Lx0 + (c + 0.5) / gx * cw

    def gj2y(r):
        return (r + 0.5) / gy * ch

    # ---- 8-neighbour grid local maxima on the smoothed field --------------
    peaks = []
    for i in range(1, gx - 1):
        for j in range(1, gy - 1):
            v = sm[i, j]
            if v <= 0:
                continue
            nb = sm[i - 1:i + 2, j - 1:j + 2]
            if v >= nb.max() and v > mean_occ:
                peaks.append((i, j, v))

    body = []
    aria = ("Density-peak prominence over a topographic relief: the projected density "
            "field rendered as colour-graded isodensity contour lines (marching squares), "
            "with grid local maxima ranked by topographic prominence against a noise cutoff.")

    # --- degrade path: no resolvable structure --------------------------------
    if len(peaks) == 0:
        for x, y in P:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.05" '
                        f'fill="var(--ink-faint)" opacity="0.5"/>')
        body.append(f'<text x="{(Lx0+Lx1)/2:.0f}" y="{H/2:.0f}" fill="var(--ink-faint)" '
                    f'font-size="11" text-anchor="middle">no resolvable density peaks</text>')
        return {"num": "DEN 04", "order": 4, "name": "Density-peak prominence",
                "tech": "peaks · prominence",
                "why": "No grid local maxima cleared the noise floor of the projected density field.",
                "svg": _svg(W, H, aria, "".join(body)),
                "legend": '<span><i class="f"></i> point (accumulates)</span>',
                "reveal": "<b>Reveals:</b> a flat, peak-free density field.",
                "cls": ""}

    # ---- topographic prominence -------------------------------------------
    # rank peaks by height; for each peak the prominence is its height minus the
    # highest saddle on any path to a strictly-higher peak (straight-line proxy).
    peaks.sort(key=lambda p: p[2], reverse=True)
    base = mean_occ  # global noise floor for the tallest peak
    proms = []
    for k, (i, j, h) in enumerate(peaks):
        if k == 0:
            key_col = base
        else:
            best_saddle = 0.0
            for (pi, pj, ph) in peaks[:k]:
                steps = max(abs(pi - i), abs(pj - j), 1)
                ts = np.linspace(0, 1, steps + 1)
                xs = np.round(i + (pi - i) * ts).astype(int)
                ys = np.round(j + (pj - j) * ts).astype(int)
                saddle = sm[xs, ys].min()
                if saddle > best_saddle:
                    best_saddle = saddle
            key_col = max(best_saddle, base)
        proms.append(max(h - key_col, 1e-6))
    proms = np.array(proms)

    rho = proms / max(base, 1e-9)          # "× diffuse-halo density"
    order = np.argsort(rho)[::-1]
    rho = rho[order]
    peaks = [peaks[o] for o in order]
    cutoff = 2.0

    # ================= LEFT PANEL: topographic relief =====================
    # faint underlying cloud (restrained ink-faint dots)
    for x, y in P:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.05" '
                    f'fill="var(--ink-faint)" opacity="0.42"/>')

    # --- isodensity levels as density quantiles of the occupied field ------
    fld = sm[occ]
    if fld.size < 8:
        fld = sm.ravel()
    qs = [0.40, 0.56, 0.70, 0.82, 0.91, 0.97]
    levels = []
    for q in qs:
        v = float(np.quantile(fld, q))
        if not levels or v > levels[-1] * 1.0001:
            levels.append(v)
    nb = len(levels)

    # colour: low->high density on a sequential token ramp; p (bad mixed into
    # accent) rises by level so the densest contour leans toward var(--bad).
    def level_color(k):
        frac = k / max(1, nb - 1)
        p = int(round(6 + frac * 80))       # 6% .. 86% bad-into-accent
        return f"color-mix(in srgb, var(--bad) {p}%, var(--accent))"

    def level_pct(k):
        frac = k / max(1, nb - 1)
        return int(round(6 + frac * 80))

    # subtle filled relief band per level (opacity <=0.12), outer (low) first
    for k, level in enumerate(levels):
        chains = _stitch(_march_segments(sm, level))
        if not chains:
            continue
        d = []
        for poly in chains:
            if len(poly) < 3:
                continue
            pts = [(gi2x(c), gj2y(r)) for (c, r) in poly]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            if (max(xs) - min(xs)) < 10 and (max(ys) - min(ys)) < 10:
                continue
            d.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z")
        if not d:
            continue
        op = 0.035 + 0.013 * k              # 0.035 .. <=0.10 — very low relief
        body.append(
            f'<path d="{" ".join(d)}" fill="{level_color(k)}" fill-opacity="{op:.3f}" '
            f'fill-rule="evenodd" stroke="none"/>'
        )

    # hairline contour polylines on top of the bands (the elevation-map look)
    for k, level in enumerate(levels):
        chains = _stitch(_march_segments(sm, level))
        if not chains:
            continue
        d = []
        for poly in chains:
            if len(poly) < 2:
                continue
            pts = [(gi2x(c), gj2y(r)) for (c, r) in poly]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            if (max(xs) - min(xs)) < 10 and (max(ys) - min(ys)) < 10:
                continue
            d.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts))
        if not d:
            continue
        sw = 0.6 + 0.08 * k
        op = 0.5 + 0.5 * (k / max(1, nb - 1))
        body.append(
            f'<path d="{" ".join(d)}" fill="none" stroke="{level_color(k)}" '
            f'stroke-opacity="{op:.2f}" stroke-width="{sw:.2f}" stroke-linejoin="round" '
            f'vector-effect="non-scaling-stroke"/>'
        )

    # ---- peak prominence markers ON TOP of the relief ---------------------
    n_named = min(int((rho >= cutoff).sum()), 3) or 1   # name up to top-3 real peaks
    rmax = float(rho.max())

    def mark_r(v):
        return 2.4 + 5.6 * (v / rmax) ** 0.6

    for k, (i, j, h) in enumerate(peaks):
        x, y = cell_xy(i, j)
        r = mark_r(rho[k])
        if k == 0:
            # dominant peak carries the accent (double-ring + filled dot)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r+3.2:.1f}" fill="none" '
                        f'stroke="var(--accent)" stroke-width="2.2" vector-effect="non-scaling-stroke"/>')
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(r,4.5):.1f}" fill="var(--accent)"/>')
            lx = x - (r + 9) if x > (Lx0 + Lx1) / 2 else x + (r + 9)
            anc = "end" if x > (Lx0 + Lx1) / 2 else "start"
            body.append(f'<line x1="{x:.1f}" y1="{y-(r+1):.1f}" x2="{lx:.1f}" y2="{y-(r+12):.1f}" '
                        f'stroke="var(--ink-faint)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
            body.append(f'<text x="{lx:.1f}" y="{y-(r+15):.1f}" fill="var(--accent)" font-size="11" '
                        f'font-weight="700" text-anchor="{anc}">P1</text>')
            body.append(f'<text x="{lx:.1f}" y="{y-(r+4):.1f}" fill="var(--ink-soft)" font-size="8.5" '
                        f'text-anchor="{anc}" style="font-variant-numeric:tabular-nums">ρ {rho[k]:.1f}×</text>')
        elif k < n_named:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(r,3.0)+1.6:.1f}" fill="var(--paper)" '
                        f'opacity="0.6"/>')
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(r,3.0):.1f}" fill="var(--ink)"/>')
            lx = x + (r + 9)
            body.append(f'<line x1="{x:.1f}" y1="{y-(r+1):.1f}" x2="{lx:.1f}" y2="{y-(r+12):.1f}" '
                        f'stroke="var(--ink-faint)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
            body.append(f'<text x="{lx:.1f}" y="{y-(r+15):.1f}" fill="var(--ink)" font-size="11" '
                        f'font-weight="700" text-anchor="start">P{k+1}</text>')
            body.append(f'<text x="{lx:.1f}" y="{y-(r+4):.1f}" fill="var(--ink-soft)" font-size="8.5" '
                        f'text-anchor="start" style="font-variant-numeric:tabular-nums">ρ {rho[k]:.1f}×</text>')
        else:
            # sub-cutoff bumps: faint hollow rings, no label clutter
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.4" fill="none" '
                        f'stroke="var(--ink-soft)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')

    # ================= RIGHT PANEL: prominence ranking =====================
    ry0, ry1 = 70.0, 452.0
    axis_max = max(np.ceil(rho.max() / 2.0) * 2.0, 4.0)

    def bx_of(v):
        return Rx0 + (v / axis_max) * (Rx1 - Rx0)

    # panel divider
    body.append(f'<line x1="{Rx0-61:.0f}" y1="58" x2="{Rx0-61:.0f}" y2="460" '
                f'stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
    # baseline (0x) and fine ticks every 2x
    body.append(f'<line x1="{Rx0:.1f}" y1="{ry0:.1f}" x2="{Rx0:.1f}" y2="{ry1:.1f}" '
                f'stroke="var(--rule)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
    body.append('<g aria-hidden="true">')
    t = 2.0
    ticks = []
    while t <= axis_max + 1e-6:
        tx = bx_of(t)
        body.append(f'<line x1="{tx:.1f}" y1="{ry0:.1f}" x2="{tx:.1f}" y2="{ry1:.1f}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
        ticks.append((tx, t))
        t += 2.0
    body.append(f'<line x1="{Rx0:.1f}" y1="{ry1:.1f}" x2="{Rx1:.1f}" y2="{ry1:.1f}" '
                f'stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
    body.append('<g fill="var(--ink-faint)" font-size="8" text-anchor="middle" '
                'style="font-variant-numeric:tabular-nums">')
    body.append(f'<text x="{Rx0:.1f}" y="{ry1+11:.1f}">0×</text>')
    for tx, tv in ticks:
        body.append(f'<text x="{tx:.1f}" y="{ry1+11:.1f}">{tv:.0f}×</text>')
    body.append('</g></g>')
    # cutoff line
    cx = bx_of(cutoff)
    body.append(f'<line x1="{cx:.1f}" y1="{ry0:.1f}" x2="{cx:.1f}" y2="{ry1:.1f}" '
                f'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3" '
                f'vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{cx+4:.1f}" y="{ry0-10:.1f}" fill="var(--ink-faint)" font-size="8.5" '
                f'style="font-variant-numeric:tabular-nums">cutoff {cutoff:.1f}×</text>')

    # one ranked row per peak (cap rows so the panel stays legible)
    show = min(len(peaks), 8)
    row_h = (ry1 - ry0) / max(show, 1)
    for k in range(show):
        yy = ry0 + (k + 0.5) * row_h
        v = rho[k]
        ex = bx_of(min(v, axis_max))
        above = v >= cutoff
        if k == 0:
            body.append(f'<line x1="{Rx0:.1f}" y1="{yy:.1f}" x2="{ex:.1f}" y2="{yy:.1f}" '
                        f'stroke="var(--accent)" stroke-width="2.2" vector-effect="non-scaling-stroke"/>')
            body.append(f'<circle cx="{ex:.1f}" cy="{yy:.1f}" r="6.1" fill="none" '
                        f'stroke="var(--accent)" stroke-width="2.2" vector-effect="non-scaling-stroke"/>')
            body.append(f'<circle cx="{ex:.1f}" cy="{yy:.1f}" r="4.5" fill="var(--accent)"/>')
            body.append(f'<text x="{Rx0-6:.1f}" y="{yy+3.5:.1f}" fill="var(--accent)" font-size="10.5" '
                        f'font-weight="700" text-anchor="end">P1</text>')
            body.append(f'<text x="{ex+9.5:.1f}" y="{yy+3.5:.1f}" fill="var(--ink-soft)" font-size="9" '
                        f'style="font-variant-numeric:tabular-nums">{v:.1f}</text>')
        elif above:
            body.append(f'<line x1="{Rx0:.1f}" y1="{yy:.1f}" x2="{ex:.1f}" y2="{yy:.1f}" '
                        f'stroke="var(--ink)" stroke-width="1.3" vector-effect="non-scaling-stroke"/>')
            body.append(f'<circle cx="{ex:.1f}" cy="{yy:.1f}" r="{2.4+1.4*(v/rmax):.1f}" fill="var(--ink)"/>')
            body.append(f'<text x="{Rx0-6:.1f}" y="{yy+3.5:.1f}" fill="var(--ink)" font-size="10.5" '
                        f'font-weight="700" text-anchor="end">P{k+1}</text>')
            body.append(f'<text x="{ex+8:.1f}" y="{yy+3.5:.1f}" fill="var(--ink-soft)" font-size="9" '
                        f'style="font-variant-numeric:tabular-nums">{v:.1f}</text>')
        else:
            body.append(f'<line x1="{Rx0:.1f}" y1="{yy:.1f}" x2="{ex:.1f}" y2="{yy:.1f}" '
                        f'stroke="var(--ink-faint)" stroke-width="1.3" stroke-dasharray="2 2" '
                        f'vector-effect="non-scaling-stroke"/>')
            body.append(f'<circle cx="{ex:.1f}" cy="{yy:.1f}" r="1.8" fill="none" '
                        f'stroke="var(--ink-faint)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
            body.append(f'<text x="{Rx0-6:.1f}" y="{yy+3.5:.1f}" fill="var(--ink-faint)" font-size="9" '
                        f'text-anchor="end">bump</text>')
            body.append(f'<text x="{ex+7:.1f}" y="{yy+3.5:.1f}" fill="var(--ink-faint)" font-size="9" '
                        f'style="font-variant-numeric:tabular-nums">{v:.1f}</text>')

    # ---- contour ramp scale on the relief panel (low->high density) -------
    sx, sy = 40, 466
    body.append('<g style="font-variant-numeric:tabular-nums">')
    body.append(f'<text x="{sx:.0f}" y="{sy-1:.0f}" fill="var(--ink-faint)" font-size="8.5" '
                f'text-anchor="start">low</text>')
    for k in range(nb):
        rx = sx + 24 + k * 18
        body.append(f'<rect x="{rx:.0f}" y="{sy-9:.0f}" width="16" height="7" '
                    f'fill="{level_color(k)}" fill-opacity="0.85" '
                    f'stroke="var(--ink-faint)" stroke-width="0.4"/>')
    body.append(f'<text x="{sx+24+nb*18+4:.0f}" y="{sy-1:.0f}" fill="var(--ink-faint)" font-size="8.5" '
                f'text-anchor="start">high density · isodensity contour ramp</text>')
    body.append('</g>')

    # ---- caption strip ----------------------------------------------------
    n_above = int((rho >= cutoff).sum())
    body.append(f'<text x="40" y="22" fill="var(--ink-soft)" font-size="11">'
                f'<tspan fill="var(--ink)" font-weight="700">{n_above} hotspot'
                f'{"s" if n_above != 1 else ""}</tspan> clear the prominence cutoff · '
                f'<tspan fill="var(--accent)" font-weight="700">P1 dominates ({rho[0]:.0f}×)</tspan>'
                f'</text>')
    body.append('<text x="40" y="50" fill="var(--ink-soft)" font-size="11">'
                'topographic relief · isodensity contours over the reservoir · peaks sized by prominence</text>')
    body.append(f'<text x="{Rx0-2:.0f}" y="50" fill="var(--ink-soft)" font-size="11">'
                'prominence ranking →</text>')
    body.append(f'<text x="{(Rx0+Rx1)/2:.0f}" y="478" fill="var(--ink-faint)" font-size="9" '
                f'text-anchor="middle">topographic prominence  (× noise-floor density)</text>')

    why = (f"The projected density field rendered as colour-graded isodensity contour lines "
           f"(marching squares) with grid local maxima ranked by topographic prominence over "
           f"the noise floor; {n_above} clear the {cutoff:.0f}× cutoff and P1 dominates at {rho[0]:.1f}×.")
    return {
        "num": "DEN 04", "order": 4, "name": "Density-peak prominence",
        "tech": "contours · peaks · prominence",
        "why": why,
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i style="background:color-mix(in srgb, var(--bad) 6%, var(--accent))"></i>'
                  ' low-density contour</span>'
                  '<span><i style="background:color-mix(in srgb, var(--bad) 86%, var(--accent))"></i>'
                  ' high-density contour</span>'
                  '<span><i class="a"></i> dominant peak (P1)</span>'
                  '<span><i class="f"></i> ranked peak / bump</span>'
                  '<span><i class="dash"></i> noise cutoff</span>',
        "reveal": "<b>Reveals:</b> the elevation map of occupied space — nested isodensity "
                  "contours show how mass concentrates, and how many genuine density modes the "
                  "dataset has; tall isolated prominence means real cluster cores, sub-cutoff "
                  "bumps are sampling noise.",
        "cls": "",
    }
