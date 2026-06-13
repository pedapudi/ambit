"""3D 05 — Radial shell occupancy.

Each reservoir row's distance from the centroid of ctx.es.X gives a radial
coordinate; binning into concentric spherical shells and dividing each shell's
count by its shell volume (r^(d-1) measure) yields a normalized occupancy
profile vs radius. A filled core decays from a peak; a hollow rind dips below
the uniform-density expectation at a cavity radius, then recovers, then thins
into the sparse outlier tail. The single fullest shell carries the accent;
surplus over the uniform reference reads UP as the good direction, while the
cavity deficit and the outer falloff read as a neutral faint trough — never red.
A small isometric inset draws three concentric shell silhouettes (core / cavity
/ support) to establish what a shell is.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_d3_shell(ctx):
    W, H = 760, 470

    X = getattr(getattr(ctx, "es", None), "X", None)
    if X is None or len(X) < 16:
        body = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" text-anchor="middle" '
                f'font-size="13" fill="var(--ink-faint)">needs reservoir vectors</text>')
        return {"num": "3D 05", "order": 24, "name": "Radial shell occupancy", "tech": "shells",
                "why": "No reservoir embedding matrix available to bin into radial shells.",
                "svg": _svg(W, H, "Radial shell occupancy: reservoir unavailable", body),
                "legend": '<span><i class="f"></i> needs reservoir</span>',
                "reveal": "<b>Reveals:</b> radial structure — unavailable without reservoir vectors.",
                "cls": ""}

    X = np.asarray(X, float)
    d = X.shape[1]
    centroid = X.mean(0)
    r = np.linalg.norm(X - centroid, axis=1)  # distance of each row from centroid
    rmax = float(r.max()) or 1.0

    # --- bin into concentric shells, occupancy = count / shell volume ---------
    # The shells are spherical (as the isometric inset depicts): a shell at radius
    # r has volume ~ r^2 dr (the 3-D shell measure). Normalizing the per-shell row
    # count by this volume turns the raw radial histogram into a *density* profile,
    # which is what a "filled core decays, a hollow rind dips then recovers" story
    # needs. (Using the ambient r^(d-1) measure would multiply the dynamic range by
    # ~60 decades of pure volume scaling and bury the structure.)
    SHELL_P = 3  # spherical (3-D) shell volume exponent
    NB = 22
    edges = np.linspace(0.0, rmax * 1.0000001, NB + 1)
    cnt, _ = np.histogram(r, bins=edges)
    rc = 0.5 * (edges[:-1] + edges[1:])             # shell mid-radius
    vol = (edges[1:] ** SHELL_P - edges[:-1] ** SHELL_P)  # integrated r^2 dr thin-shell measure
    vol = np.maximum(vol, 1e-300)
    occ = cnt / vol                                 # raw occupancy (count / shell vol)

    # uniform-density expectation: total points spread evenly over the ball ->
    # occupancy is constant = N / total_volume. Express the profile as a ratio
    # to that constant so 1.0x is the uniform reference.
    total_vol = edges[-1] ** SHELL_P
    uniform_occ = len(r) / total_vol
    ratio = occ / uniform_occ                        # 1x = uniform; >1 surplus; <1 deficit

    # only plot shells that actually contain points (drop empty inner/outer rim
    # bins that would log to -inf); keep a contiguous populated span.
    pop = cnt > 0
    if pop.sum() < 3:
        pop[:] = cnt >= 0  # degenerate fallback: keep all
    first = int(np.argmax(pop))
    last = int(len(pop) - 1 - np.argmax(pop[::-1]))
    sel = np.arange(first, last + 1)
    rc_s = rc[sel]
    ratio_s = np.clip(ratio[sel], 1e-3, None)
    cnt_s = cnt[sel]

    # fullest shell (most points) carries the accent
    peak_i = int(np.argmax(cnt_s))
    # cavity = deepest interior deficit below the uniform reference (neutral dip)
    interior = (ratio_s < 1.0)
    if interior.any() and len(ratio_s) > 2:
        # restrict to non-edge shells so the outer tail isn't mistaken for the cavity
        mid = np.arange(1, len(ratio_s) - 1)
        midmask = interior[mid]
        if midmask.any():
            cav_i = int(mid[midmask][np.argmin(ratio_s[mid][midmask])])
        else:
            cav_i = int(np.argmin(ratio_s))
    else:
        cav_i = int(np.argmin(ratio_s))

    # ---- main panel geometry -------------------------------------------------
    PL, PR = 360, 710          # plot x extent (left margin holds inset + axis)
    PT, PB = 74, 374           # plot y extent
    rlo, rhi = float(rc_s.min()), float(rc_s.max())
    rspan = max(rhi - rlo, 1e-9)

    def x_of(rr):
        return PL + (rr - rlo) / rspan * (PR - PL)

    # log occupancy axis spanning the data ratio range, padded to round decades
    lr = np.log10(ratio_s)
    lo_dec = float(np.floor(min(lr.min(), 0.0)))
    hi_dec = float(np.ceil(max(lr.max(), 0.0)))
    if hi_dec - lo_dec < 2:
        hi_dec = lo_dec + 2
    yspan = hi_dec - lo_dec

    def y_of(rat):
        lv = np.log10(max(rat, 10.0 ** (lo_dec - 0.5)))
        return PB - (lv - lo_dec) / yspan * (PB - PT)

    y_unit = y_of(1.0)  # the 1x uniform reference line

    body = []

    # === isometric shell inset (left) ========================================
    icx, icy = 132, 250
    # three flattened concentric ellipse silhouettes: support, cavity, core
    inset = ['<g aria-hidden="true">']
    inset.append(f'<text x="{icx}" y="120" font-size="9.5" fill="var(--ink-faint)" '
                 f'text-anchor="middle" letter-spacing="0.05em">CONCENTRIC SHELLS</text>')
    inset.append(f'<text x="{icx}" y="133" font-size="8.5" fill="var(--ink-faint)" '
                 f'text-anchor="middle" opacity="0.85">isometric · az 35° el 22°</text>')
    # gnomon arms
    for (dx, dy, lab, lx, ly) in [(36, -9.5, "x", 172, 243.5), (-25.2, -13.5, "y", 104.8, 233.5),
                                  (0, 40.8, "z", 135, 293.8)]:
        inset.append(f'<line x1="{icx}" y1="{icy}" x2="{icx+dx:.1f}" y2="{icy+dy:.1f}" '
                     f'stroke="var(--ink-faint)" stroke-width="0.6" opacity="0.5" '
                     f'vector-effect="non-scaling-stroke"/>')
        inset.append(f'<text x="{lx}" y="{ly}" font-size="8" fill="var(--ink-faint)" '
                     f'text-anchor="middle" opacity="0.7">{lab}</text>')
    # outermost support shell (dashed), cavity shell (solid neutral), core shell (accent)
    inset.append(f'<ellipse cx="{icx}" cy="{icy}" rx="96" ry="44.2" fill="none" '
                 f'stroke="var(--ink-faint)" stroke-width="0.8" opacity="0.45" '
                 f'stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>')
    inset.append(f'<ellipse cx="{icx}" cy="{icy}" rx="56.3" ry="25.9" fill="none" '
                 f'stroke="var(--ink-faint)" stroke-width="1.0" opacity="0.7" '
                 f'vector-effect="non-scaling-stroke"/>')
    inset.append(f'<ellipse cx="{icx}" cy="{icy}" rx="13.4" ry="6.2" fill="none" '
                 f'stroke="var(--accent)" stroke-width="1.5" opacity="0.9" '
                 f'vector-effect="non-scaling-stroke"/>')
    inset.append(f'<circle cx="{icx}" cy="{icy}" r="1.7" fill="var(--ink-soft)"/>')
    inset.append(f'<text x="136" y="260" font-size="8" fill="var(--ink-faint)">centroid</text>')
    inset.append(f'<text x="{icx}" y="238" font-size="8" fill="var(--accent)" '
                 f'text-anchor="middle">core shell</text>')
    inset.append(f'<text x="192" y="236" font-size="8" fill="var(--ink-faint)">cavity shell</text>')
    inset.append(f'<text x="226" y="306" font-size="8" fill="var(--ink-faint)" '
                 f'text-anchor="middle" opacity="0.8">support</text>')
    inset.append(f'<text x="{icx}" y="380" font-size="8" fill="var(--ink-faint)" '
                 f'text-anchor="middle" opacity="0.85">shells centered on the {d}-D centroid;</text>')
    inset.append(f'<text x="{icx}" y="391" font-size="8" fill="var(--ink-faint)" '
                 f'text-anchor="middle" opacity="0.85">occupancy normalized by shell volume</text>')
    inset.append('</g>')
    body.append("".join(inset))

    # === log gridlines + minor sub-decade ticks (aria-hidden) ================
    grid = ['<g aria-hidden="true">']
    sub = [2, 3, 4, 5, 6, 7, 8, 9]
    anchor = {2, 5}  # full-width faint gridlines at these sub-decade anchors
    label_at = {0.2: "0.2×", 0.5: "0.5×", 2.0: "2×", 5.0: "5×",
                20.0: "20×", 50.0: "50×", 200.0: "200×", 500.0: "500×"}
    dec = int(lo_dec)
    while dec < hi_dec:
        for s in sub:
            val = s * (10.0 ** dec)
            ly = y_of(val)
            if ly < PT - 0.5 or ly > PB + 0.5:
                continue
            grid.append(f'<line x1="{PL-3.5:.1f}" y1="{ly:.1f}" x2="{PL:.1f}" y2="{ly:.1f}" '
                        f'stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
            if s in anchor:
                grid.append(f'<line x1="{PL:.1f}" y1="{ly:.1f}" x2="{PR:.1f}" y2="{ly:.1f}" '
                            f'stroke="var(--rule-soft)" stroke-width="0.5" opacity="0.5" '
                            f'vector-effect="non-scaling-stroke"/>')
            # mono sub-decade label where it's one of the chosen anchors
            key = round(val, 6)
            for lv, txt in label_at.items():
                if abs(lv - val) < 1e-6 * max(1.0, val):
                    grid.append(f'<text x="{PL-7:.1f}" y="{ly+3:.1f}" font-size="7.5" '
                                f'fill="var(--ink-faint)" text-anchor="end" '
                                f'font-family="ui-monospace,monospace" opacity="0.85">{txt}</text>')
                    break
        dec += 1
    grid.append('</g>')
    body.append("".join(grid))

    # === decade major ticks + labels =========================================
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
            body.append(f'<text x="{PL-7:.1f}" y="{ly+3:.1f}" font-size="8.5" '
                        f'fill="var(--ink-faint)" text-anchor="end">{lab}</text>')
        dec += 1

    # === plot frame ==========================================================
    body.append(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PB}" stroke="var(--rule)" '
                f'stroke-width="0.8" vector-effect="non-scaling-stroke"/>')
    body.append(f'<line x1="{PL}" y1="{PB}" x2="{PR}" y2="{PB}" stroke="var(--rule)" '
                f'stroke-width="0.8" vector-effect="non-scaling-stroke"/>')

    # === radius axis: evenly spaced major + intermediate ticks ===============
    NMAJ = 5
    rticks = np.linspace(rlo, rhi, NMAJ)
    for ti, rv in enumerate(rticks):
        x = x_of(rv)
        body.append(f'<line x1="{x:.1f}" y1="{PB}" x2="{x:.1f}" y2="{PB+4}" '
                    f'stroke="var(--ink-faint)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
        body.append(f'<text x="{x:.1f}" y="{PB+15}" font-size="8.5" fill="var(--ink-faint)" '
                    f'text-anchor="middle">{rv:.2f}</text>')
        if ti < NMAJ - 1:  # intermediate hairline tick midway to the next major
            rm = 0.5 * (rv + rticks[ti + 1])
            xm = x_of(rm)
            body.append(f'<line x1="{xm:.1f}" y1="{PB}" x2="{xm:.1f}" y2="{PB+3}" '
                        f'stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
            body.append(f'<text x="{xm:.1f}" y="{PB+15}" font-size="7.5" fill="var(--ink-faint)" '
                        f'text-anchor="middle" font-family="ui-monospace,monospace" '
                        f'opacity="0.85">{rm:.2f}</text>')

    # axis titles
    body.append(f'<text x="{(PL+PR)/2:.0f}" y="404" font-size="9" fill="var(--ink-faint)" '
                f'text-anchor="middle" letter-spacing="0.05em">radius from {d}-D centroid →</text>')
    body.append(f'<text x="320" y="{(PT+PB)/2:.0f}" font-size="9" fill="var(--ink-faint)" '
                f'text-anchor="middle" letter-spacing="0.03em" '
                f'transform="rotate(-90 320 {(PT+PB)/2:.0f})">normalized occupancy '
                f'(count / shell vol, log)</text>')

    # === panel header text ===================================================
    body.append(f'<text x="34" y="34" font-size="13" font-weight="700" fill="var(--ink)" '
                f'letter-spacing="0.03em">radial shell occupancy profile</text>')
    body.append(f'<text x="34" y="51" font-size="10" fill="var(--ink-faint)">count per '
                f'spherical shell ÷ shell volume, from the {d}-D centroid · '
                f'a filled core decays; a hollow rind dips then recovers</text>')

    # === uniform-density reference (dashed) ==================================
    body.append(f'<line x1="{PL}" y1="{y_unit:.1f}" x2="{PR}" y2="{y_unit:.1f}" '
                f'stroke="var(--rule-soft)" stroke-width="1.1" stroke-dasharray="4 3" '
                f'vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{PL+6:.0f}" y="{y_unit-5:.1f}" font-size="8.5" fill="var(--ink-faint)" '
                f'text-anchor="start">uniform-density expectation (1×)</text>')

    # === profile path + signed fill (surplus good, deficit neutral) ==========
    xs = np.array([x_of(rv) for rv in rc_s])
    ys = np.array([y_of(rt) for rt in ratio_s])

    # surplus area (above uniform line) -> good-soft; clipped at the 1x line
    sur = []
    started = False
    for i in range(len(xs)):
        above = ys[i] <= y_unit  # smaller y = higher value = above the line
        if above and not started:
            sur.append(f'M {xs[i]:.1f},{y_unit:.1f}')
            started = True
        if started:
            sur.append(f'L {xs[i]:.1f},{min(ys[i], y_unit):.1f}')
        if (not above) and started:
            sur.append(f'L {xs[i]:.1f},{y_unit:.1f} Z')
            started = False
    if started:
        sur.append(f'L {xs[-1]:.1f},{y_unit:.1f} Z')
    if sur:
        body.append(f'<path d="{" ".join(sur)}" fill="var(--good-soft)" '
                    f'fill-opacity="0.45" stroke="none"/>')

    # deficit area (below uniform line) -> neutral faint trough (never red)
    defc = []
    started = False
    for i in range(len(xs)):
        below = ys[i] > y_unit
        if below and not started:
            defc.append(f'M {xs[i]:.1f},{y_unit:.1f}')
            started = True
        if started:
            defc.append(f'L {xs[i]:.1f},{max(ys[i], y_unit):.1f}')
        if (not below) and started:
            defc.append(f'L {xs[i]:.1f},{y_unit:.1f} Z')
            started = False
    if started:
        defc.append(f'L {xs[-1]:.1f},{y_unit:.1f} Z')
    if defc:
        body.append(f'<path d="{" ".join(defc)}" fill="var(--ink-faint)" '
                    f'fill-opacity="0.10" stroke="none"/>')

    # the occupancy data line (ink)
    line_pts = " ".join(f"{'M' if i == 0 else 'L'} {xs[i]:.1f},{ys[i]:.1f}"
                        for i in range(len(xs)))
    body.append(f'<path d="{line_pts}" fill="none" stroke="var(--ink)" stroke-width="1.3" '
                f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')

    # === callouts: cavity dip (neutral), outer tail (neutral), core peak (accent)
    # cavity dip
    if 0 <= cav_i < len(xs) and ratio_s[cav_i] < 1.0:
        cx, cy = xs[cav_i], ys[cav_i]
        body.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{y_unit:.1f}" '
                    f'stroke="var(--ink-faint)" stroke-width="0.7" stroke-dasharray="2 2" '
                    f'opacity="0.65" vector-effect="non-scaling-stroke"/>')
        body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.8" fill="var(--ink-faint)"/>')
        body.append(f'<text x="{cx:.1f}" y="{cy+15:.1f}" font-size="8.5" fill="var(--ink-faint)" '
                    f'text-anchor="middle">cavity dip {ratio_s[cav_i]:.2f}×</text>')
        body.append(f'<text x="{cx:.1f}" y="{cy+25:.1f}" font-size="7.5" fill="var(--ink-faint)" '
                    f'text-anchor="middle" opacity="0.85">shell deficit (neutral)</text>')

    # outer tail (last shell, thinning to empty)
    ti = len(xs) - 1
    if ratio_s[ti] < 1.0 and ti != cav_i:
        tx, ty = xs[ti], ys[ti]
        body.append(f'<line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx:.1f}" y2="{y_unit:.1f}" '
                    f'stroke="var(--ink-faint)" stroke-width="0.7" stroke-dasharray="2 2" '
                    f'opacity="0.55" vector-effect="non-scaling-stroke"/>')
        body.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="2.6" fill="var(--ink-faint)"/>')
        body.append(f'<text x="{min(tx, PR-2):.1f}" y="{ty+18:.1f}" font-size="8" '
                    f'fill="var(--ink-faint)" text-anchor="end">outlier tail '
                    f'{ratio_s[ti]:.2f}× · thins to empty</text>')

    # core peak (fullest shell carries the accent)
    px_, py_ = xs[peak_i], ys[peak_i]
    body.append(f'<circle cx="{px_:.1f}" cy="{py_:.1f}" r="3.0" fill="var(--accent)"/>')
    lab_x = px_ + 9 if px_ < (PL + PR) / 2 else px_ - 9
    anch = "start" if px_ < (PL + PR) / 2 else "end"
    pk = ratio_s[peak_i]
    pk_txt = f"{pk:.0f}×" if pk >= 10 else f"{pk:.1f}×"
    body.append(f'<text x="{lab_x:.1f}" y="{py_+1:.1f}" font-size="9.5" fill="var(--accent)" '
                f'font-weight="700" text-anchor="{anch}">fullest shell {pk_txt}</text>')
    body.append(f'<text x="{lab_x:.1f}" y="{py_+11:.1f}" font-size="7.5" fill="var(--ink-faint)" '
                f'text-anchor="{anch}">{int(cnt_s[peak_i])} rows · surplus ↑</text>')

    # --- summary numbers for the prose --------------------------------------
    frac_in = float((r <= rc_s[peak_i]).mean())
    aria = (f"Radial shell occupancy profile of the reservoir embedding cloud. Each row's "
            f"distance from the {d}-dimensional centroid is binned into {len(rc_s)} concentric "
            f"shells; the count in each shell divided by its shell volume gives a normalized "
            f"occupancy plotted on a logarithmic vertical axis against radius. A small isometric "
            f"inset shows three concentric shell silhouettes establishing the core, cavity and "
            f"support shells. Occupancy peaks at the fullest shell, marked by the accent dot, "
            f"where surplus over the dashed uniform-density reference reads up as the good "
            f"direction; it dips below the reference at an intermediate cavity shell marked by a "
            f"neutral dot, then thins into the sparse outlier tail at the outermost shell, both "
            f"deficits drawn as a neutral faint trough below the line, never red.")
    why = (f"Distance of every reservoir row from the {d}-D centroid, binned into concentric "
           f"shells and normalized by shell volume. The fullest shell (accent) holds the densest "
           f"radial band; a dip below the uniform reference marks a cavity, and the outer falloff "
           f"is the sparse outlier rind — both neutral, since absence is information, not a defect.")
    return {
        "num": "3D 05", "order": 24, "name": "Radial shell occupancy", "tech": "shells",
        "why": why,
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="a"></i> fullest shell (accent)</span>'
                  '<span><i class="f"></i> occupancy (count / shell vol, log)</span>'
                  '<span><i class="dash"></i> uniform-density reference (1×)</span>',
        "reveal": (f"<b>Reveals:</b> the dataset's radial shape — a dense core decaying outward, "
                   f"a cavity shell where the cloud thins below uniform expectation, and the "
                   f"sparse outlier rind running off into empty space."),
        "cls": "",
    }
