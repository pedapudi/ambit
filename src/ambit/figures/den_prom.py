from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_den_prom(ctx):
    """Density-peak prominence: grid local maxima of the projected density field,
    ranked by topographic prominence. Each peak is a dot over the faint cloud sized
    by its prominence; the single dominant peak carries the accent. A fine prominence
    axis on the right ranks every peak against a noise cutoff."""
    W, H = 900, 488
    # left panel = cloud + sited peaks ; right panel = prominence ranking
    Lx0, Lx1 = 30, 530          # cloud box
    Rx0, Rx1 = 606, 818         # ranking bar field
    pad = 24

    # ---- project the reservoir into the cloud box -------------------------
    P = _box(ctx.xy, Lx1 - Lx0, H, pad=pad)
    P[:, 0] += Lx0              # shift into left panel

    # ---- density field on a grid, then find grid local maxima -------------
    gx, gy = 56, 40
    cw, ch = (Lx1 - Lx0), H
    bx = np.clip(((P[:, 0] - Lx0) / cw * gx).astype(int), 0, gx - 1)
    by = np.clip((P[:, 1] / ch * gy).astype(int), 0, gy - 1)
    grid = np.zeros((gx, gy))
    np.add.at(grid, (bx, by), 1)
    # light smoothing (3x3 box) so single-cell spikes don't dominate
    sm = grid.copy()
    for ax in (0, 1):
        sm = (np.roll(sm, 1, ax) + sm + np.roll(sm, -1, ax))
    sm /= 9.0
    occ = sm > 0
    mean_occ = sm[occ].mean() if occ.any() else 1.0

    # 8-neighbour local maxima on the smoothed grid
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
    # header text (mirrors template)
    aria = ("Density-peak prominence: grid local maxima of the projected density "
            "field, ranked by topographic prominence with a noise cutoff.")

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
    # highest saddle on any path to a strictly-higher peak. Approximate the saddle
    # by a descending flood: prominence = h - max over higher peaks of the
    # minimum-cell-on-straight-line (a robust, cheap topographic proxy).
    peaks.sort(key=lambda p: p[2], reverse=True)
    base = mean_occ  # global noise floor for the tallest peak
    proms = []
    for k, (i, j, h) in enumerate(peaks):
        if k == 0:
            key_col = base
        else:
            # min density along straight grid lines to each higher peak; the
            # best (highest) such saddle is the key col separating this peak
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

    # normalise prominence to a multiple of the noise floor for a readable scale
    rho = proms / max(base, 1e-9)          # "× diffuse-halo density"
    order = np.argsort(rho)[::-1]
    rho = rho[order]
    peaks = [peaks[o] for o in order]
    cutoff = 2.0                            # peaks above this clear the noise

    # grid cell -> svg coords (cell centre)
    def cell_xy(i, j):
        return (Lx0 + (i + 0.5) / gx * cw, (j + 0.5) / gy * ch)

    # ================= LEFT PANEL: cloud + sited peaks =====================
    # faint cloud (restrained ink-faint dots, density by accumulation)
    for x, y in P:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.05" '
                    f'fill="var(--ink-faint)" opacity="0.5"/>')

    n_named = min(int((rho >= cutoff).sum()), 3) or 1   # name up to top-3 real peaks
    # marker radius scales with prominence (r ~ 2..8)
    rmax = float(rho.max())

    def mark_r(v):
        return 2.4 + 5.6 * (v / rmax) ** 0.6

    for k, (i, j, h) in enumerate(peaks):
        x, y = cell_xy(i, j)
        r = mark_r(rho[k])
        above = rho[k] >= cutoff
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
            body.append(f'<text x="{lx:.1f}" y="{y-(r+4):.1f}" fill="var(--ink-faint)" font-size="8.5" '
                        f'text-anchor="{anc}" style="font-variant-numeric:tabular-nums">ρ {rho[k]:.1f}×</text>')
        elif k < n_named:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(r,3.0):.1f}" fill="var(--ink)"/>')
            lx = x + (r + 9)
            body.append(f'<line x1="{x:.1f}" y1="{y-(r+1):.1f}" x2="{lx:.1f}" y2="{y-(r+12):.1f}" '
                        f'stroke="var(--ink-faint)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
            body.append(f'<text x="{lx:.1f}" y="{y-(r+15):.1f}" fill="var(--ink)" font-size="11" '
                        f'font-weight="700" text-anchor="start">P{k+1}</text>')
            body.append(f'<text x="{lx:.1f}" y="{y-(r+4):.1f}" fill="var(--ink-faint)" font-size="8.5" '
                        f'text-anchor="start" style="font-variant-numeric:tabular-nums">ρ {rho[k]:.1f}×</text>')
        else:
            # sub-cutoff bumps: faint hollow rings, no label clutter
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.4" fill="none" '
                        f'stroke="var(--ink-faint)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')

    # ================= RIGHT PANEL: prominence ranking =====================
    ry0, ry1 = 70.0, 452.0
    # x-axis scale: 0 .. axis_max in "×" units
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
            # dominant peak: accent bar + ring marker
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
            # sub-cutoff bump: faint dashed bar
            body.append(f'<line x1="{Rx0:.1f}" y1="{yy:.1f}" x2="{ex:.1f}" y2="{yy:.1f}" '
                        f'stroke="var(--ink-faint)" stroke-width="1.3" stroke-dasharray="2 2" '
                        f'vector-effect="non-scaling-stroke"/>')
            body.append(f'<circle cx="{ex:.1f}" cy="{yy:.1f}" r="1.8" fill="none" '
                        f'stroke="var(--ink-faint)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
            body.append(f'<text x="{Rx0-6:.1f}" y="{yy+3.5:.1f}" fill="var(--ink-faint)" font-size="9" '
                        f'text-anchor="end">bump</text>')
            body.append(f'<text x="{ex+7:.1f}" y="{yy+3.5:.1f}" fill="var(--ink-faint)" font-size="9" '
                        f'style="font-variant-numeric:tabular-nums">{v:.1f}</text>')

    # ---- caption strip ----------------------------------------------------
    n_above = int((rho >= cutoff).sum())
    body.append(f'<text x="40" y="22" fill="var(--ink-soft)" font-size="11">'
                f'<tspan fill="var(--ink)" font-weight="700">{n_above} hotspot'
                f'{"s" if n_above != 1 else ""}</tspan> clear the prominence cutoff · '
                f'<tspan fill="var(--accent)" font-weight="700">P1 dominates ({rho[0]:.0f}×)</tspan>'
                f'</text>')
    body.append('<text x="40" y="50" fill="var(--ink-soft)" font-size="11">'
                'density field · grid local maxima sited on the reservoir · sized by prominence</text>')
    body.append(f'<text x="{Rx0-2:.0f}" y="50" fill="var(--ink-soft)" font-size="11">'
                'prominence ranking →</text>')
    body.append(f'<text x="{(Rx0+Rx1)/2:.0f}" y="478" fill="var(--ink-faint)" font-size="9" '
                f'text-anchor="middle">topographic prominence  (× noise-floor density)</text>')

    why = (f"Local maxima of the projected density field, ranked by topographic "
           f"prominence over the noise floor; {n_above} clear the {cutoff:.0f}× cutoff and "
           f"P1 dominates at {rho[0]:.1f}×.")
    return {
        "num": "DEN 04", "order": 4, "name": "Density-peak prominence",
        "tech": "peaks · prominence",
        "why": why,
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="a"></i> dominant peak (P1)</span>'
                  '<span><i class="f"></i> ranked peak / bump</span>'
                  '<span><i class="dash"></i> noise cutoff</span>',
        "reveal": "<b>Reveals:</b> how many genuine density modes the dataset has and how "
                  "dominant the largest is — tall isolated prominence means real cluster cores; "
                  "sub-cutoff bumps are sampling noise.",
        "cls": "",
    }
