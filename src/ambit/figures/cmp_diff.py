"""CMP 10 — Differential vs isotropic reference (signed log-ratio occupancy field).

Grid the bounding box of the projected reservoir, compare the dataset's per-cell
occupancy to a matched isotropic reference (an axis-matched Gaussian over the same
projected scatter), and render the signed log2 ratio as a diverging accounting
heatmap: surplus (var--good) where this dataset crowds a cell beyond reference,
deficit (var--bad) where it abandons reference-populated space, parity transparent.
A dashed L=0 iso-contour rings the structure; both spatial axes carry standardized
projection-unit ticks and a numbered ratio colorbar states the scale.
"""
from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_cmp_diff(ctx):
    w, h = 760, 690
    # ---- plot box (square cells) ------------------------------------------------
    x0, y0 = 60.0, 60.0          # top-left of the field
    fw, fh = 540.0, 540.0        # field is square -> square cells
    G = 30                        # cells per side
    cw = fw / G                   # cell pixel size (square)

    xy = np.asarray(ctx.xy, float)
    # standardize to projection units (mean 0, unit std per axis) so the grid and
    # the isotropic reference share one coordinate frame; clip to +/-3 sigma.
    mu = xy.mean(0)
    sd = xy.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (xy - mu) / sd
    R = 3.0  # half-extent of the field in standardized units (-3 .. +3)

    # cell index of each point (only points inside the +/-R window populate cells)
    gx = np.floor((Z[:, 0] + R) / (2 * R) * G).astype(int)
    gy = np.floor((Z[:, 1] + R) / (2 * R) * G).astype(int)
    inside = (gx >= 0) & (gx < G) & (gy >= 0) & (gy < G)
    counts = np.zeros((G, G))
    np.add.at(counts, (gx[inside], gy[inside]), 1.0)
    n_in = counts.sum()

    # ---- matched isotropic reference -------------------------------------------
    # The reference is an axis-matched isotropic Gaussian over the SAME standardized
    # frame: each cell's reference mass is the analytic Gaussian density at the cell
    # centre. This is the "what an isotropic blob would put here" baseline, so the
    # log-ratio isolates structure the dataset adds or removes vs. an even spread.
    cc = (np.arange(G) + 0.5) / G * (2 * R) - R          # cell-centre std units
    cx, cy = np.meshgrid(cc, cc, indexing="ij")
    ref = np.exp(-0.5 * (cx ** 2 + cy ** 2))             # isotropic Gaussian kernel
    ref = ref / ref.sum()

    p_data = counts / max(n_in, 1.0)
    eps = 0.5 / max(n_in, 1.0)                            # one half-count floor
    L = np.log2((p_data + eps) / (ref + eps))            # signed log-ratio per cell
    Lcap = 3.0                                            # ramp saturates at +/-3
    Lc = np.clip(L, -Lcap, Lcap)

    # only cells that carry data OR meaningful reference mass are part of the field
    ref_floor = ref.max() * 0.02
    active = (counts > 0) | (ref > ref_floor)

    # ---- helpers ----------------------------------------------------------------
    def cell_px(i, j):
        # i along x, j along y; svg y grows downward so flip j
        px = x0 + i * cw
        py = y0 + (G - 1 - j) * cw
        return px, py

    def u_to_px_x(u):  # standardized unit -> svg x
        return x0 + (u + R) / (2 * R) * fw

    def u_to_px_y(u):  # standardized unit -> svg y (flipped)
        return y0 + (R - u) / (2 * R) * fh

    body = []
    # header subtitles -----------------------------------------------------------
    body.append(f'<text x="{x0:.0f}" y="34" fill="var(--ink-soft)" font-size="12">'
                'signed log-ratio field · L = log₂((p_data+ε)/(p_ref+ε)) '
                'vs matched isotropic reference</text>')
    body.append(f'<text x="{x0:.0f}" y="48" fill="var(--ink-faint)" font-size="10">'
                f'{G}×{G} square cells · same projection as every figure · '
                'colour = occupancy accounting (surplus↔deficit), not a resolution score</text>')

    # ---- diverging cell fills ---------------------------------------------------
    for i in range(G):
        for j in range(G):
            px, py = cell_px(i, j)
            if not active[i, j]:
                # empty reference-and-data region: faint grid stub, no fill
                body.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{cw:.2f}" '
                            f'height="{cw:.2f}" fill="none" stroke="var(--rule-soft)" '
                            f'stroke-width="0.5" shape-rendering="crispEdges"/>')
                continue
            v = Lc[i, j]
            mag = abs(v) / Lcap            # 0..1
            pct = int(round(7 + mag * 93))  # 7%..100% so parity reads near-transparent
            if v >= 0:
                fill = f'color-mix(in srgb, var(--good) {pct}%, transparent)'
            else:
                fill = f'color-mix(in srgb, var(--bad) {pct}%, transparent)'
            if abs(v) < 1e-3:
                fill = 'color-mix(in srgb, var(--rule-soft) 12%, transparent)'
            body.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{cw:.2f}" '
                        f'height="{cw:.2f}" fill="{fill}" shape-rendering="crispEdges"/>')

    # ---- L=0 parity iso-contour (marching-squares edge crossings) --------------
    # mark where the sign of L flips between horizontally/vertically adjacent active
    # cells -> draw a short dashed segment on that shared edge.
    seg = []
    for i in range(G):
        for j in range(G):
            if not active[i, j]:
                continue
            px, py = cell_px(i, j)
            # right edge
            if i + 1 < G and active[i + 1, j] and (L[i, j] >= 0) != (L[i + 1, j] >= 0):
                ex = px + cw
                seg.append((ex, py, ex, py + cw))
            # top edge (smaller j is lower in data -> j+1 is upper, py smaller)
            if j + 1 < G and active[i, j + 1] and (L[i, j] >= 0) != (L[i, j + 1] >= 0):
                ey = py
                seg.append((px, ey, px + cw, ey))
    for (a, b, c, dd) in seg:
        body.append(f'<line x1="{a:.2f}" y1="{b:.2f}" x2="{c:.2f}" y2="{dd:.2f}" '
                    'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3" '
                    'vector-effect="non-scaling-stroke"/>')

    # ---- field frame ------------------------------------------------------------
    body.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{fw:.2f}" height="{fh:.2f}" '
                'fill="none" stroke="var(--rule)" stroke-width="1" '
                'vector-effect="non-scaling-stroke"/>')

    # ---- spatial axes: standardized projection-unit ticks ----------------------
    ax_b = y0 + fh
    tick = []
    for u in range(-3, 4):
        xx = u_to_px_x(u)
        tick.append(f'<line x1="{xx:.2f}" y1="{ax_b:.2f}" x2="{xx:.2f}" y2="{ax_b+6:.2f}" '
                    'stroke="var(--ink-faint)" stroke-width="1" vector-effect="non-scaling-stroke"/>')
        tick.append(f'<text x="{xx:.2f}" y="{ax_b+17:.2f}" fill="var(--ink-faint)" '
                    'font-size="9" text-anchor="middle" '
                    f'style="font-variant-numeric:tabular-nums">{u:+d}</text>'.replace('+0', '0'))
        yy = u_to_px_y(u)
        tick.append(f'<line x1="{x0-6:.2f}" y1="{yy:.2f}" x2="{x0:.2f}" y2="{yy:.2f}" '
                    'stroke="var(--ink-faint)" stroke-width="1" vector-effect="non-scaling-stroke"/>')
        tick.append(f'<text x="{x0-9:.2f}" y="{yy+3:.2f}" fill="var(--ink-faint)" '
                    'font-size="9" text-anchor="end" '
                    f'style="font-variant-numeric:tabular-nums">{u:+d}</text>'.replace('+0', '0'))
    # half-unit minor ticks + hairline gridlines
    for k in range(-6, 7):
        u = k * 0.5
        xx = u_to_px_x(u)
        yy = u_to_px_y(u)
        tick.append(f'<line x1="{xx:.2f}" y1="{y0:.2f}" x2="{xx:.2f}" y2="{ax_b:.2f}" '
                    'stroke="var(--rule-soft)" stroke-width="0.4" '
                    'vector-effect="non-scaling-stroke"/>')
        tick.append(f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x0+fw:.2f}" y2="{yy:.2f}" '
                    'stroke="var(--rule-soft)" stroke-width="0.4" '
                    'vector-effect="non-scaling-stroke"/>')
        if k % 2 != 0:  # half-unit minor tick marks
            tick.append(f'<line x1="{xx:.2f}" y1="{ax_b:.2f}" x2="{xx:.2f}" y2="{ax_b+3:.2f}" '
                        'stroke="var(--rule-soft)" stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
            tick.append(f'<line x1="{x0-3:.2f}" y1="{yy:.2f}" x2="{x0:.2f}" y2="{yy:.2f}" '
                        'stroke="var(--rule-soft)" stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
    body.append(f'<g shape-rendering="crispEdges">{"".join(tick)}</g>')
    body.append(f'<text x="{x0+fw/2:.2f}" y="{ax_b+30:.2f}" fill="var(--ink-soft)" '
                'font-size="9.5" text-anchor="middle">projection x · standardized units</text>')
    body.append(f'<text x="{x0-34:.2f}" y="{y0+fh/2:.2f}" fill="var(--ink-soft)" '
                f'font-size="9.5" text-anchor="middle" transform="rotate(-90 {x0-34:.2f} {y0+fh/2:.2f})">'
                'projection y · standardized units</text>')

    # ---- vertical colorbar: numbered ratio scale -------------------------------
    bx = 626.0
    bw = 14.0
    by_top = y0          # +Lcap at top
    bh = fh
    steps = 60
    seg_h = bh / steps
    bar = []
    for s in range(steps):
        # top = +Lcap, bottom = -Lcap
        v = Lcap - (s + 0.5) / steps * (2 * Lcap)
        mag = abs(v) / Lcap
        pct = int(round(7 + mag * 93))
        if v >= 0:
            f = f'color-mix(in srgb, var(--good) {pct}%, transparent)'
        else:
            f = f'color-mix(in srgb, var(--bad) {pct}%, transparent)'
        bar.append(f'<rect x="{bx:.1f}" y="{by_top + s*seg_h:.2f}" width="{bw:.0f}" '
                   f'height="{seg_h+0.6:.2f}" fill="{f}" shape-rendering="crispEdges"/>')
    body.append("".join(bar))
    body.append(f'<rect x="{bx:.1f}" y="{by_top:.1f}" width="{bw:.0f}" height="{bh:.1f}" '
                'fill="none" stroke="var(--rule-soft)" stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
    # ratio scale labels: L value and the equivalent fold-ratio
    for v in (3, 2, 1, 0, -1, -2, -3):
        yy = by_top + (Lcap - v) / (2 * Lcap) * bh
        col = 'var(--good)' if v > 0 else ('var(--bad)' if v < 0 else 'var(--ink-soft)')
        body.append(f'<line x1="{bx+bw:.1f}" y1="{yy:.1f}" x2="{bx+bw+4:.1f}" y2="{yy:.1f}" '
                    'stroke="var(--ink-faint)" stroke-width="1" vector-effect="non-scaling-stroke"/>')
        lab = f'{v:+d}' if v != 0 else ' 0'
        body.append(f'<text x="{bx+bw+6:.1f}" y="{yy+3:.1f}" fill="{col}" font-size="9" '
                    f'text-anchor="start" style="font-variant-numeric:tabular-nums">{lab}</text>')
        # fold ratio sub-label (2^L) on the inner side
        fold = 2.0 ** abs(v)
        if v != 0:
            sub = f'×{fold:.0f}' if fold >= 2 else ''
            body.append(f'<text x="{bx-3:.1f}" y="{yy+3:.1f}" fill="var(--ink-faint)" '
                        f'font-size="7.5" text-anchor="end" '
                        f'style="font-variant-numeric:tabular-nums">{sub}</text>')
    body.append(f'<text x="{bx+bw/2:.1f}" y="{by_top-8:.1f}" fill="var(--good)" font-size="8" '
                'text-anchor="middle">surplus</text>')
    body.append(f'<text x="{bx+bw/2:.1f}" y="{by_top+bh+12:.1f}" fill="var(--bad)" font-size="8" '
                'text-anchor="middle">deficit</text>')
    body.append(f'<text x="{bx+bw/2:.1f}" y="{by_top+bh+22:.1f}" fill="var(--ink-faint)" '
                'font-size="7.5" text-anchor="middle">L = log₂ ratio</text>')

    # ---- bottom caption ---------------------------------------------------------
    body.append(f'<text x="{x0:.0f}" y="{ax_b+52:.0f}" fill="var(--ink-faint)" font-size="9.5">'
                'occupancy accounting (not a resolution score): '
                'green = this dataset crowds a cell beyond the isotropic reference, '
                'red = reference-populated space this dataset abandons; parity is transparent.</text>')

    aria = ("Differential log-ratio occupancy field: signed surplus and deficit of the "
            "dataset against a matched isotropic reference, square-cell grid with an "
            "L=0 parity iso-contour and standardized projection-unit axes; the colour "
            "ramp is an occupancy accounting axis (green surplus, red deficit), not a "
            "resolution good/bad score")

    # ---- headline numbers for the why/reveal -----------------------------------
    if active.any():
        Lact = L[active]
        peak_pos = float(Lact.max())
        peak_neg = float(Lact.min())
    else:
        peak_pos = peak_neg = 0.0
    n_surplus = int(((L > 0.25) & active).sum())
    n_deficit = int(((L < -0.25) & active).sum())

    legend = (
        '<span><i style="background:var(--good)"></i> surplus L&gt;0 — occupancy above the '
        'isotropic reference (this dataset crowds the cell beyond the base)</span>'
        '<span><i style="background:var(--bad)"></i> deficit L&lt;0 — occupancy below the '
        'reference (reference-populated space this dataset abandons)</span>'
        '<span><i style="background:transparent;border-color:var(--rule-soft)"></i> '
        'parity L≈0 — matches the reference</span>'
        '<span><i class="dash"></i> L=0 parity boundary (dashed)</span>'
        f'<span class="leg-note">numbered ratio scale: L=±1 is a 2× fold, L=±2 a '
        f'4× fold, L=±3 an 8× fold · accounting axis, not a resolution good/bad '
        'score</span>')

    reveal = (
        '<b>Reveals:</b> <b>where the dataset over- vs under-occupies</b> its embedding '
        'space relative to an even isotropic spread — an occupancy accounting, not a '
        f'quality verdict. The densest structure reads as the largest <span class="good">'
        f'surplus</span> (peak L≈{peak_pos:+.1f}, up to a {2.0**abs(peak_pos):.0f}× '
        f'fold over reference), while the slack the reference would fill reads as '
        f'<span class="bad">deficit</span> (down to L≈{peak_neg:+.1f}). '
        f'{n_surplus} cells run surplus, {n_deficit} run deficit; the dashed '
        '<span class="faint">L=0 boundary</span> rings the occupied structure and parity '
        'with the reference stays a transparent neutral midpoint. Green and red mark '
        'occupancy surplus and deficit, not resolution.')

    return {
        "num": "CMP 10", "order": 10,
        "name": "Differential vs isotropic reference", "tech": "log-ratio",
        "why": ("Per-cell occupancy of the projected reservoir against a matched isotropic "
                "reference, as a signed log₂ ratio. Surplus (green) is where this dataset "
                "crowds beyond an even spread; deficit (red) is reference space it abandons; "
                "parity is transparent."),
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": legend,
        "reveal": reveal,
        "cls": "",
    }
