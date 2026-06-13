"""3D 05 — Radial shell occupancy.

Each reservoir row's distance from the centroid of ctx.es.X gives a high-dim
radial coordinate; binning by radius quantiles into ~6 concentric shells (each
holding a comparable count) sorts every row into a shell from the filled core to
the sparse rind. The PRIMARY panel makes the shells visible *in 3-D*: every row
is placed on its nested sphere — direction taken from the unit-normalised 3-D
projection ctx.xyz, radius taken from its normalised high-dim distance — and the
cloud is drawn in a single fixed ISOMETRIC pose (azimuth 35°, elevation 22°),
points coloured by shell on a core->rind sequential ramp and ordered by depth
(painter's, far z first) so the layering reads as genuine 3-D structure. The
concentric shell boundaries are drawn as isometric ELLIPSES — the equatorial
circle of each shell sphere projected through the same camera — over a centroid
dot and small x/y/z axis hints. A SECONDARY smaller panel keeps the
occupancy-vs-radius profile: count per spherical shell over shell volume, where
surplus over the uniform-density reference reads UP as the good direction, the
cavity dip reads neutral, and the single fullest shell carries the accent.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_d3_shell(ctx):
    W, H = 760, 470

    X = getattr(getattr(ctx, "es", None), "X", None)
    xyz = getattr(ctx, "xyz", None)
    if X is None or len(X) < 16 or xyz is None:
        body = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" text-anchor="middle" '
                f'font-size="13" fill="var(--ink-faint)">needs reservoir vectors</text>')
        return {"num": "3D 05", "order": 24, "name": "Radial shell occupancy", "tech": "shells",
                "why": "No reservoir embedding matrix available to bin into radial shells.",
                "svg": _svg(W, H, "Radial shell occupancy: reservoir unavailable", body),
                "legend": '<span><i class="f"></i> needs reservoir</span>',
                "reveal": "<b>Reveals:</b> radial structure — unavailable without reservoir vectors.",
                "cls": ""}

    X = np.asarray(X, float)
    xyz = np.asarray(xyz, float)
    d = X.shape[1]
    centroid = X.mean(0)
    r = np.linalg.norm(X - centroid, axis=1)          # high-dim distance per row
    rmin, rmax = float(r.min()), float(r.max())
    rspan_r = max(rmax - rmin, 1e-12)
    rnorm = (r - rmin) / rspan_r                       # normalised to [0,1]

    # --- quantile shells: ~6 concentric shells, each ~equal count -------------
    NS = 6
    qs = np.quantile(r, np.linspace(0.0, 1.0, NS + 1))
    qs[0] = rmin
    qs[-1] = rmax * 1.0000001
    # assign each row to a shell index 0..NS-1 by its radius
    shell = np.clip(np.searchsorted(qs, r, side="right") - 1, 0, NS - 1)
    # outer boundary radius of each shell (the ring we will draw)
    bound_r = qs[1:]                                   # length NS
    bound_rnorm = (bound_r - rmin) / rspan_r           # normalised outer radius

    # === binning for the SECONDARY occupancy profile (count / shell volume) ===
    SHELL_P = 3                                        # spherical 3-D shell measure
    NB = 22
    edges = np.linspace(0.0, rmax * 1.0000001, NB + 1)
    cnt, _ = np.histogram(r, bins=edges)
    rc = 0.5 * (edges[:-1] + edges[1:])
    vol = (edges[1:] ** SHELL_P - edges[:-1] ** SHELL_P)
    vol = np.maximum(vol, 1e-300)
    occ = cnt / vol
    total_vol = edges[-1] ** SHELL_P
    uniform_occ = len(r) / total_vol
    ratio = occ / uniform_occ                          # 1x = uniform; >1 surplus

    pop = cnt > 0
    if pop.sum() < 3:
        pop[:] = cnt >= 0
    first = int(np.argmax(pop))
    last = int(len(pop) - 1 - np.argmax(pop[::-1]))
    sel = np.arange(first, last + 1)
    rc_s = rc[sel]
    ratio_s = np.clip(ratio[sel], 1e-3, None)
    cnt_s = cnt[sel]
    peak_i = int(np.argmax(cnt_s))
    interior = (ratio_s < 1.0)
    if interior.any() and len(ratio_s) > 2:
        mid = np.arange(1, len(ratio_s) - 1)
        midmask = interior[mid]
        cav_i = int(mid[midmask][np.argmin(ratio_s[mid][midmask])]) if midmask.any() \
            else int(np.argmin(ratio_s))
    else:
        cav_i = int(np.argmin(ratio_s))

    body = []

    # =====================================================================
    # PRIMARY PANEL — 3-D ISOMETRIC concentric shells (the point cloud on
    # nested spheres, drawn in a single fixed camera pose)
    # =====================================================================
    cx0, cy0 = 210, 250        # screen centre of the isometric scene (left panel)
    S = 168.0                  # world-unit -> screen scale (unit sphere -> radius S)

    # --- shared isometric camera (azimuth 35°, elevation 22°) -----------------
    az = np.deg2rad(35.0)
    el = np.deg2rad(22.0)
    caz, saz = np.cos(az), np.sin(az)
    cel, sel_ = np.cos(el), np.sin(el)

    def project(px, py, pz):
        """World (x,y,z) -> (screen_x, screen_y, depth). Larger depth = nearer."""
        x1 = px * caz + pz * saz
        z1 = -px * saz + pz * caz
        y1 = py * cel - z1 * sel_
        sx = cx0 + x1 * S
        sy = cy0 - y1 * S
        return sx, sy, z1            # z1 ~ camera depth (use for painter's order)

    # --- place every row on its nested sphere ---------------------------------
    # direction = unit(ctx.xyz); radius = normalised high-dim distance in [0,1]
    nrm = np.linalg.norm(xyz, axis=1)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    dirv = xyz / nrm[:, None]
    rscene = rnorm                                     # 0..1 radius on the unit sphere
    P3 = dirv * rscene[:, None]                        # (n,3) world positions

    # subsample to ~3500 for a clean isometric cloud
    n = len(r)
    NMAX = 3500
    if n > NMAX:
        rng = np.random.default_rng(7)
        idx = rng.choice(n, NMAX, replace=False)
    else:
        idx = np.arange(n)

    sx, sy, depth = project(P3[idx, 0], P3[idx, 1], P3[idx, 2])
    sh = shell[idx]

    # core->rind sequential token ramp: shell p% rises per shell.
    # core (shell 0) = pure accent; rind (shell NS-1) = mostly bad/ink-faint.
    def shell_fill(s):
        p = int(round(15 + (s / max(NS - 1, 1)) * 70))   # 15% -> 85%
        return f"color-mix(in srgb, var(--bad) {p}%, var(--accent))"

    # header
    body.append(f'<text x="34" y="34" font-size="13" font-weight="700" fill="var(--ink)" '
                f'letter-spacing="0.03em">concentric shells</text>')
    body.append(f'<text x="34" y="51" font-size="9.5" fill="var(--ink-faint)">points on '
                f'nested spheres by distance from the {d}-D centroid · isometric '
                f'(az 35° · el 22°) · colour = shell (core → rind)</text>')

    # --- axis hints at the scene centre (small x/y/z ticks) -------------------
    AX = 0.30                                          # axis hint length in world units
    axes = ['<g aria-hidden="true">']
    for vec, lab in (((AX, 0, 0), "x"), ((0, AX, 0), "y"), ((0, 0, AX), "z")):
        ax, ay, _ = project(vec[0], vec[1], vec[2])
        axes.append(f'<line x1="{cx0:.1f}" y1="{cy0:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
                    f'stroke="var(--ink-faint)" stroke-width="0.7" opacity="0.7" '
                    f'vector-effect="non-scaling-stroke"/>')
        lx, ly = cx0 + (ax - cx0) * 1.12, cy0 + (ay - cy0) * 1.12
        axes.append(f'<text x="{lx:.1f}" y="{ly+2.5:.1f}" font-size="7.5" '
                    f'fill="var(--ink-faint)" text-anchor="middle" opacity="0.75" '
                    f'font-family="ui-monospace,monospace">{lab}</text>')
    axes.append('</g>')
    body.append("".join(axes))

    # --- the points, drawn in painter's order (far z first) -------------------
    # sort the whole subsample by depth so nearer points overpaint farther ones,
    # then emit one <g> per shell colour preserving that depth order.
    order = np.argsort(depth)                           # ascending: far -> near
    sx_o, sy_o, sh_o = sx[order], sy[order], sh[order]
    pts_g = ['<g aria-hidden="true">']
    # bucket by shell but keep depth order within: stream through depth order,
    # grouping consecutive runs of the same shell into a coloured <g>.
    fills = [shell_fill(s) for s in range(NS)]
    i = 0
    m = len(order)
    while i < m:
        s = int(sh_o[i])
        j = i
        seg = []
        while j < m and int(sh_o[j]) == s:
            seg.append(f'<circle cx="{sx_o[j]:.1f}" cy="{sy_o[j]:.1f}" r="1.3"/>')
            j += 1
        op = 0.6 if s == 0 else 0.78
        pts_g.append(f'<g fill="{fills[s]}" fill-opacity="{op}">{"".join(seg)}</g>')
        i = j
    pts_g.append('</g>')
    body.append("".join(pts_g))

    # --- concentric shell-boundary ELLIPSES -----------------------------------
    # The equatorial circle of each shell sphere is the circle of radius R in the
    # world x-z plane (y = 0). Projected through the camera it is an ellipse; we
    # sample it as a polyline so it stays exact under the isometric transform.
    NTH = 96
    th = np.linspace(0.0, 2.0 * np.pi, NTH + 1)
    cth, snth = np.cos(th), np.sin(th)
    label_shells = {1, 3, NS - 1}
    rings = ['<g aria-hidden="true">']
    for s in range(NS):
        R = float(bound_rnorm[s])
        ex, ey, _ = project(R * cth, np.zeros_like(cth), R * snth)
        # faint dark under-stroke for legibility over the dense cloud, then the
        # dashed hairline ellipse on top.
        ptsstr = " ".join(f"{ex[k]:.1f},{ey[k]:.1f}" for k in range(len(ex)))
        rings.append(f'<polyline points="{ptsstr}" fill="none" stroke="var(--panel)" '
                     f'stroke-width="1.7" opacity="0.5" vector-effect="non-scaling-stroke"/>')
        rings.append(f'<polyline points="{ptsstr}" fill="none" stroke="var(--ink-faint)" '
                     f'stroke-width="0.8" stroke-dasharray="2 3" opacity="0.85" '
                     f'vector-effect="non-scaling-stroke"/>')
        if s in label_shells:
            # label at the near (bottom) extremum of the ellipse
            lk = int(np.argmax(ey))
            rings.append(f'<text x="{ex[lk]:.1f}" y="{ey[lk]+9:.1f}" font-size="7.5" '
                         f'fill="var(--ink-faint)" text-anchor="middle" '
                         f'font-family="ui-monospace,monospace" opacity="0.95" '
                         f'paint-order="stroke" stroke="var(--panel)" stroke-width="2.4">'
                         f'{bound_r[s]:.2f}</text>')
    rings.append('</g>')
    body.append("".join(rings))

    # centroid marker at the scene centre
    body.append(f'<circle cx="{cx0}" cy="{cy0}" r="2.0" fill="var(--ink-soft)"/>')
    body.append(f'<text x="{cx0}" y="{cy0+14:.0f}" font-size="7.5" fill="var(--ink-faint)" '
                f'text-anchor="middle" opacity="0.8" paint-order="stroke" '
                f'stroke="var(--panel)" stroke-width="2.2">centroid</text>')

    # shell colour legend strip beneath the isometric scene (core -> rind)
    lg_x0, lg_y = 110, 452
    lg_w = 200.0
    body.append(f'<text x="{lg_x0:.0f}" y="{lg_y-9:.0f}" font-size="8" fill="var(--ink-faint)" '
                f'letter-spacing="0.04em">core</text>')
    body.append(f'<text x="{lg_x0+lg_w:.0f}" y="{lg_y-9:.0f}" font-size="8" '
                f'fill="var(--ink-faint)" text-anchor="end" letter-spacing="0.04em">rind</text>')
    for s in range(NS):
        sw = lg_w / NS
        body.append(f'<rect x="{lg_x0 + s*sw:.1f}" y="{lg_y-7:.1f}" width="{sw+0.6:.1f}" '
                    f'height="7" fill="{shell_fill(s)}" fill-opacity="0.85"/>')

    # =====================================================================
    # SECONDARY PANEL — occupancy-vs-radius profile (smaller, right)
    # =====================================================================
    PL, PR = 470, 712          # plot x extent
    PT, PB = 96, 360           # plot y extent
    rlo, rhi = float(rc_s.min()), float(rc_s.max())
    pspan = max(rhi - rlo, 1e-9)

    def x_of(rr):
        return PL + (rr - rlo) / pspan * (PR - PL)

    lr = np.log10(ratio_s)
    lo_dec = float(np.floor(min(lr.min(), 0.0)))
    hi_dec = float(np.ceil(max(lr.max(), 0.0)))
    if hi_dec - lo_dec < 2:
        hi_dec = lo_dec + 2
    yspan = hi_dec - lo_dec

    def y_of(rat):
        lv = np.log10(max(rat, 10.0 ** (lo_dec - 0.5)))
        return PB - (lv - lo_dec) / yspan * (PB - PT)

    y_unit = y_of(1.0)

    # secondary header
    body.append(f'<text x="{PL:.0f}" y="{PT-30:.0f}" font-size="10.5" font-weight="700" '
                f'fill="var(--ink)" letter-spacing="0.02em">occupancy vs radius</text>')
    body.append(f'<text x="{PL:.0f}" y="{PT-17:.0f}" font-size="8" fill="var(--ink-faint)">'
                f'count / shell volume, log · surplus ↑ good</text>')

    # log gridlines + sub-decade ticks
    grid = ['<g aria-hidden="true">']
    anchor = {2, 5}
    label_at = {0.2: "0.2×", 0.5: "0.5×", 2.0: "2×", 5.0: "5×",
                20.0: "20×", 50.0: "50×", 200.0: "200×", 500.0: "500×"}
    dec = int(lo_dec)
    while dec < hi_dec:
        for s in [2, 3, 4, 5, 6, 7, 8, 9]:
            val = s * (10.0 ** dec)
            ly = y_of(val)
            if ly < PT - 0.5 or ly > PB + 0.5:
                continue
            grid.append(f'<line x1="{PL-3:.1f}" y1="{ly:.1f}" x2="{PL:.1f}" y2="{ly:.1f}" '
                        f'stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
            if s in anchor:
                grid.append(f'<line x1="{PL:.1f}" y1="{ly:.1f}" x2="{PR:.1f}" y2="{ly:.1f}" '
                            f'stroke="var(--rule-soft)" stroke-width="0.5" opacity="0.5" '
                            f'vector-effect="non-scaling-stroke"/>')
            for lv, txt in label_at.items():
                if abs(lv - val) < 1e-6 * max(1.0, val):
                    grid.append(f'<text x="{PL-6:.1f}" y="{ly+3:.1f}" font-size="7" '
                                f'fill="var(--ink-faint)" text-anchor="end" '
                                f'font-family="ui-monospace,monospace" opacity="0.85">{txt}</text>')
                    break
        dec += 1
    grid.append('</g>')
    body.append("".join(grid))

    # decade major ticks
    dec = int(lo_dec)
    while dec <= hi_dec:
        val = 10.0 ** dec
        ly = y_of(val)
        if PT - 0.5 <= ly <= PB + 0.5:
            body.append(f'<line x1="{PL-4:.1f}" y1="{ly:.1f}" x2="{PL:.1f}" y2="{ly:.1f}" '
                        f'stroke="var(--ink-faint)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
            if dec == 0:
                lab = "1×"
            elif dec == 1:
                lab = "10×"
            elif dec < 0:
                lab = f"{val:g}×"
            else:
                lab = f"{int(val)}×"
            body.append(f'<text x="{PL-7:.1f}" y="{ly+3:.1f}" font-size="8" '
                        f'fill="var(--ink-faint)" text-anchor="end">{lab}</text>')
        dec += 1

    # frame
    body.append(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PB}" stroke="var(--rule)" '
                f'stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
    body.append(f'<line x1="{PL}" y1="{PB}" x2="{PR}" y2="{PB}" stroke="var(--rule)" '
                f'stroke-width="0.8" vector-effect="non-scaling-stroke"/>')

    # radius axis ticks
    NMAJ = 4
    rticks = np.linspace(rlo, rhi, NMAJ)
    for rv in rticks:
        x = x_of(rv)
        body.append(f'<line x1="{x:.1f}" y1="{PB}" x2="{x:.1f}" y2="{PB+4}" '
                    f'stroke="var(--ink-faint)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
        body.append(f'<text x="{x:.1f}" y="{PB+14}" font-size="8" fill="var(--ink-faint)" '
                    f'text-anchor="middle">{rv:.2f}</text>')
    body.append(f'<text x="{(PL+PR)/2:.0f}" y="{PB+27:.0f}" font-size="8.5" '
                f'fill="var(--ink-faint)" text-anchor="middle" letter-spacing="0.04em">'
                f'radius from {d}-D centroid →</text>')

    # uniform-density reference (dashed)
    body.append(f'<line x1="{PL}" y1="{y_unit:.1f}" x2="{PR}" y2="{y_unit:.1f}" '
                f'stroke="var(--rule-soft)" stroke-width="1.0" stroke-dasharray="4 3" '
                f'vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{PR-2:.0f}" y="{y_unit-4:.1f}" font-size="7.5" fill="var(--ink-faint)" '
                f'text-anchor="end">uniform (1×)</text>')

    xs = np.array([x_of(rv) for rv in rc_s])
    ys = np.array([y_of(rt) for rt in ratio_s])

    # surplus area (above uniform) -> good-soft
    sur, started = [], False
    for i in range(len(xs)):
        above = ys[i] <= y_unit
        if above and not started:
            sur.append(f'M {xs[i]:.1f},{y_unit:.1f}'); started = True
        if started:
            sur.append(f'L {xs[i]:.1f},{min(ys[i], y_unit):.1f}')
        if (not above) and started:
            sur.append(f'L {xs[i]:.1f},{y_unit:.1f} Z'); started = False
    if started:
        sur.append(f'L {xs[-1]:.1f},{y_unit:.1f} Z')
    if sur:
        body.append(f'<path d="{" ".join(sur)}" fill="var(--good-soft)" '
                    f'fill-opacity="0.45" stroke="none"/>')

    # deficit area (below uniform) -> neutral faint trough
    defc, started = [], False
    for i in range(len(xs)):
        below = ys[i] > y_unit
        if below and not started:
            defc.append(f'M {xs[i]:.1f},{y_unit:.1f}'); started = True
        if started:
            defc.append(f'L {xs[i]:.1f},{max(ys[i], y_unit):.1f}')
        if (not below) and started:
            defc.append(f'L {xs[i]:.1f},{y_unit:.1f} Z'); started = False
    if started:
        defc.append(f'L {xs[-1]:.1f},{y_unit:.1f} Z')
    if defc:
        body.append(f'<path d="{" ".join(defc)}" fill="var(--ink-faint)" '
                    f'fill-opacity="0.10" stroke="none"/>')

    # occupancy line
    line_pts = " ".join(f"{'M' if i == 0 else 'L'} {xs[i]:.1f},{ys[i]:.1f}"
                        for i in range(len(xs)))
    body.append(f'<path d="{line_pts}" fill="none" stroke="var(--ink)" stroke-width="1.2" '
                f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')

    # cavity dip (neutral)
    if 0 <= cav_i < len(xs) and ratio_s[cav_i] < 1.0:
        ccx, ccy = xs[cav_i], ys[cav_i]
        body.append(f'<circle cx="{ccx:.1f}" cy="{ccy:.1f}" r="2.4" fill="var(--ink-faint)"/>')
        body.append(f'<text x="{ccx:.1f}" y="{ccy+13:.1f}" font-size="7.5" fill="var(--ink-faint)" '
                    f'text-anchor="middle">cavity {ratio_s[cav_i]:.2f}×</text>')

    # core peak (accent)
    px_, py_ = xs[peak_i], ys[peak_i]
    body.append(f'<circle cx="{px_:.1f}" cy="{py_:.1f}" r="2.8" fill="var(--accent)"/>')
    lab_x = px_ + 8 if px_ < (PL + PR) / 2 else px_ - 8
    anch = "start" if px_ < (PL + PR) / 2 else "end"
    pk = ratio_s[peak_i]
    pk_txt = f"{pk:.0f}×" if pk >= 10 else f"{pk:.1f}×"
    body.append(f'<text x="{lab_x:.1f}" y="{py_+1:.1f}" font-size="8.5" fill="var(--accent)" '
                f'font-weight="700" text-anchor="{anch}">fullest {pk_txt}</text>')

    # --- summary numbers / prose ---------------------------------------------
    frac_in = float((r <= rc_s[peak_i]).mean())
    aria = (f"Concentric shells of the reservoir embedding cloud, drawn in 3-D. The primary panel "
            f"places every reservoir row on a nested sphere — its direction taken from the 3-D "
            f"projection and its radius from its normalized distance to the {d}-dimensional centroid "
            f"— and renders the cloud in a single fixed isometric camera pose at azimuth 35 degrees "
            f"and elevation 22 degrees, with nearer points painted over farther ones. Each point is "
            f"coloured by which of {NS} radius-quantile shells it falls in, on a sequential ramp from "
            f"the accent at the dense core to a faint hue at the sparse rind. Hairline dashed "
            f"ellipses are the equatorial circles of each shell sphere projected through the same "
            f"camera, a few labelled with the radius value, over a centroid dot and small x, y and z "
            f"axis hints. A smaller secondary panel plots occupancy, the count per spherical shell "
            f"divided by shell volume, on a logarithmic axis against radius; surplus over the dashed "
            f"uniform-density reference reads up as the good direction, the fullest shell carries the "
            f"accent dot, and an intermediate cavity dip is marked neutral.")
    why = (f"Distance of every reservoir row from the {d}-D centroid sorts it into one of {NS} "
           f"concentric radius-quantile shells; the isometric view places each point on its nested "
           f"sphere and colours it by shell so the core-to-rind layering is visible directly in 3-D, "
           f"while the occupancy profile shows the fullest shell (accent), a neutral cavity dip below "
           f"the uniform reference, and the sparse outlier rind — absence is information, not a defect.")
    return {
        "num": "3D 05", "order": 24, "name": "Radial shell occupancy", "tech": "shells",
        "why": why,
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="a"></i> fullest shell / core (accent)</span>'
                  '<span><i class="f"></i> shell ramp (core → rind)</span>'
                  '<span><i class="dash"></i> shell boundary ellipse · uniform reference</span>',
        "reveal": (f"<b>Reveals:</b> which radial shell each point lives in, in 3-D — a dense accent "
                   f"core on the inner spheres, concentric shells layering outward, a cavity where the "
                   f"cloud thins below uniform expectation, and the sparse outlier rind on the "
                   f"outermost sphere running into empty space."),
        "cls": "",
    }
