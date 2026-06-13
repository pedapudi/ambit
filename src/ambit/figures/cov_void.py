"""COV 08 — Void detection via empty discs.

Find the largest point-free interior discs in the projected reservoir: a fine
grid of distance-to-nearest-point, top non-overlapping maxima. Each void is
flagged NEUTRALLY in var(--accent) (absence is not a defect), the single
largest promoted to a bolder accent spine. Radii labelled in hull units.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_cov_void(ctx):
    W, H, PAD = 900, 560, 28
    PLOT_B = H - 44  # leave a strip below the cloud for the axis

    xy = getattr(ctx, "xy", None)
    if xy is None or len(xy) < 8:
        body = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" text-anchor="middle" '
                f'font-size="13" fill="var(--ink-faint)">needs xy projection</text>')
        return {"num": "COV 08", "order": 8, "name": "Void detection", "tech": "empty discs",
                "why": "No projected reservoir available to scan for voids.",
                "svg": _svg(W, H, "Void detection: projection unavailable", body),
                "legend": '<span><i class="f"></i> needs xy</span>',
                "reveal": "<b>Reveals:</b> coverage gaps — unavailable without a 2-D projection.",
                "cls": ""}

    # --- map the reservoir into the cloud sub-box (above the axis strip) -----
    P = _box(xy, W, PLOT_B + PAD, pad=PAD)  # fit into [PAD, W-PAD] x [PAD, PLOT_B]
    px, py = P[:, 0], P[:, 1]
    x0, x1 = px.min(), px.max()
    y0, y1 = py.min(), py.max()
    diag = float(np.hypot(x1 - x0, y1 - y0)) or 1.0
    # "hull units": diameter of the point cloud's projected extent -> 0..200 scale
    hull_diam_px = diag
    px_per_unit = hull_diam_px / 200.0 if hull_diam_px else 1.0

    # --- distance-to-nearest-point field on a fine grid ---------------------
    GX, GY = 120, 90
    gx = np.linspace(x0, x1, GX)
    gy = np.linspace(y0, y1, GY)
    GXX, GYY = np.meshgrid(gx, gy)  # (GY, GX)
    cells = np.column_stack([GXX.ravel(), GYY.ravel()])  # (Ncell, 2)

    # nearest data-point distance per grid cell, chunked to bound memory
    nd = np.empty(len(cells), float)
    step = 2048
    for s in range(0, len(cells), step):
        c = cells[s:s + step]
        d2 = ((c[:, None, 0] - px[None, :]) ** 2 +
              (c[:, None, 1] - py[None, :]) ** 2)
        nd[s:s + step] = np.sqrt(d2.min(1))
    dist = nd.reshape(GY, GX)

    # interior mask: only consider cells comfortably inside the cloud's bbox so
    # we name genuine interior voids, not the empty margin around the hull.
    inset_x = 0.06 * (x1 - x0)
    inset_y = 0.06 * (y1 - y0)
    interior = ((GXX >= x0 + inset_x) & (GXX <= x1 - inset_x) &
                (GYY >= y0 + inset_y) & (GYY <= y1 - inset_y))
    distm = np.where(interior, dist, -1.0)

    # --- greedy non-overlapping top maxima (empty discs) --------------------
    order = np.argsort(distm.ravel())[::-1]
    flat_x = GXX.ravel()
    flat_y = GYY.ravel()
    flat_r = distm.ravel()
    voids = []  # (cx, cy, r)
    max_voids = 8
    for idx in order:
        r = flat_r[idx]
        if r <= 0 or not np.isfinite(r):
            break
        cx, cy = flat_x[idx], flat_y[idx]
        # reject if it overlaps an already-selected, larger void
        ok = True
        for (vx, vy, vr) in voids:
            if np.hypot(cx - vx, cy - vy) < 0.85 * (vr + r):
                ok = False
                break
        if ok:
            voids.append((float(cx), float(cy), float(r)))
        if len(voids) >= max_voids:
            break

    # discard discs that are basically pixel noise (smaller than a tick of scale)
    if voids:
        rmax = max(v[2] for v in voids)
        voids = [v for v in voids if v[2] >= 0.18 * rmax]
    voids.sort(key=lambda v: v[2], reverse=True)

    body = []

    # --- faint cloud (restrained scatter, accumulation reads density) -------
    dens = _local_density(P, W, PLOT_B + PAD)
    dmax = float(dens.max()) or 1.0
    dots = []
    for i in range(len(P)):
        op = 0.30 + 0.22 * (dens[i] / dmax)  # denser cells slightly more opaque
        dots.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="1.3" '
                    f'fill="var(--ink-faint)" opacity="{op:.2f}"/>')
    body.append('<g aria-hidden="true">' + "".join(dots) + '</g>')

    # --- void discs (neutral var(--accent); absence flagged, not condemned) -
    spine = voids[0] if voids else None
    disc_g = []
    for rank, (cx, cy, r) in enumerate(voids):
        is_spine = (rank == 0)
        sw = 2.4 if is_spine else 1.3
        fop = 0.10 if is_spine else 0.055
        dash = '' if is_spine else ' stroke-dasharray="4 3"'
        disc_g.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="var(--accent)" '
            f'fill-opacity="{fop:.3f}" stroke="var(--accent)" stroke-width="{sw}"{dash} '
            f'vector-effect="non-scaling-stroke"/>')
        # center crosshair
        disc_g.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.6" fill="var(--accent)"/>')
        # radius leader + label, in hull units
        ru = r / px_per_unit
        lx = cx + r * 0.7071
        ly = cy - r * 0.7071
        disc_g.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" '
            f'stroke="var(--accent)" stroke-width="0.8" stroke-dasharray="2 2" '
            f'opacity="0.7" vector-effect="non-scaling-stroke"/>')
        tcol = "var(--accent)"
        fs = 11 if is_spine else 9.5
        tag = f"Ø{ru:.0f}u" if not is_spine else f"largest void Ø{ru:.0f}u"
        ty = ly - 3 if ly > PAD + 14 else ly + 12
        disc_g.append(
            f'<text font-family="var(--mono, ui-monospace, monospace)" x="{lx:.1f}" '
            f'y="{ty:.1f}" fill="{tcol}" font-size="{fs}" text-anchor="middle" '
            f'style="font-variant-numeric:tabular-nums">{tag}</text>')
    body.append('<g>' + "".join(disc_g) + '</g>')

    # --- fine quantitative axis (hull units) along the bottom ---------------
    ax_y = PLOT_B + 14
    ax_x0, ax_x1 = PAD, W - PAD
    axis = [f'<line x1="{ax_x0}" y1="{ax_y}" x2="{ax_x1}" y2="{ax_y}" '
            f'stroke="var(--rule-soft)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>']
    span_px = ax_x1 - ax_x0
    span_units = span_px / px_per_unit
    # choose a round tick step (~10 ticks)
    raw = span_units / 10.0
    mag = 10 ** np.floor(np.log10(max(raw, 1e-6)))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            stepu = m * mag
            break
    else:
        stepu = 10 * mag
    nticks = int(np.floor(span_units / stepu)) + 1
    for t in range(nticks):
        u = t * stepu
        x = ax_x0 + u * px_per_unit
        if x > ax_x1 + 0.5:
            break
        major = (t % 2 == 0)
        tl = 6 if major else 3
        axis.append(f'<line x1="{x:.1f}" y1="{ax_y}" x2="{x:.1f}" y2="{ax_y+tl}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
        if major:
            axis.append(
                f'<text font-family="var(--mono, ui-monospace, monospace)" x="{x:.1f}" '
                f'y="{ax_y+18:.1f}" fill="var(--ink-faint)" font-size="8.5" '
                f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{u:.0f}</text>')
    axis.append(
        f'<text font-family="var(--mono, ui-monospace, monospace)" x="{ax_x0:.1f}" '
        f'y="{ax_y-6:.1f}" fill="var(--ink-faint)" font-size="8.5" text-anchor="start" '
        f'opacity="0.85">hull units (Ø hull ≈ 200 u)</text>')
    body.append('<g aria-hidden="true">' + "".join(axis) + '</g>')

    if not voids:
        body.append(
            f'<text x="{W/2:.0f}" y="{PLOT_B/2:.0f}" text-anchor="middle" font-size="12" '
            f'fill="var(--ink-faint)">no resolvable interior voids</text>')

    nfound = len(voids)
    big_u = (voids[0][2] / px_per_unit) if voids else 0.0
    aria = (f"Void detection: the {nfound} largest point-free interior discs in the "
            f"projected cloud, flagged neutrally; the largest spans about "
            f"{big_u:.0f} hull units.")
    why = (f"The {nfound} largest empty interior discs in the projection — regions the "
           f"dataset never visits. Voids are flagged neutrally (absence is information, "
           f"not a defect); the largest is promoted to the accent spine.")
    return {
        "num": "COV 08", "order": 8, "name": "Void detection", "tech": "empty discs",
        "why": why,
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="f"></i> reservoir point</span>'
                  '<span><i class="a"></i> empty disc (void)</span>'
                  '<span><i class="dash"></i> radius (hull units)</span>',
        "reveal": "<b>Reveals:</b> coverage gaps — large interior discs the dataset "
                  "leaves unpopulated, sized in hull units of the projected extent.",
        "cls": "",
    }
