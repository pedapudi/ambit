from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_d3_trip(ctx):
    # ---- geometry from ctx ------------------------------------------------
    # Three orthographic projections of the same 3-D PCA reservoir, drawn at
    # ONE identical numeric scale so the thin-in-Z anisotropy is measurable by
    # comparing how far the cloud fills the box in each plane.
    W, H = 752, 258
    PW = W // 3                      # panel width (≈250)
    pad = 26
    top, bot = 40, H - 34            # plot band inside each panel
    bL, bR = pad, PW - 10           # plot band left/right inside each panel

    xyz = np.asarray(getattr(ctx, "xyz", None), float)
    if xyz is None or xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) == 0:
        note = ('<text x="%d" y="%d" font-size="12" fill="var(--ink-faint)" '
                'text-anchor="middle">needs 3-D projection (ctx.xyz)</text>'
                % (W // 2, H // 2))
        return {
            "num": "3D 02", "order": 21, "name": "Orthographic triptych",
            "tech": "xy · xz · yz",
            "why": "Three orthographic projections of the 3-D reservoir at one shared scale; the 3-D projection was unavailable for this run.",
            "svg": _svg(W, H, "Orthographic triptych unavailable: no 3-D projection", note),
            "legend": '<span><i class="f"></i> cloud point</span>',
            "reveal": "<b>Reveals:</b> nothing yet — the 3-D projection (ctx.xyz) is missing.",
            "cls": "",
        }

    X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Symmetric, fixed scale shared by all three panels. Round the half-extent
    # up to a 0.2 major so the axis lands on labelled majors and every panel is
    # plotted at the identical data->pixel ratio.
    half = float(np.abs(xyz[:, :3]).max())
    SCALE = float(np.ceil(half / 0.2) * 0.2)          # e.g. 0.9 -> 1.0
    if SCALE <= 0:
        SCALE = 0.2
    LO, HI = -SCALE, SCALE

    def sx(v):  # data value -> x px within a panel (shared mapping)
        return bL + (v - LO) / (HI - LO) * (bR - bL)

    def sy(v):  # data value -> y px (flipped)
        return bot - (v - LO) / (HI - LO) * (bot - top)

    # standard deviation along each axis quantifies the anisotropy in the prose
    sdx, sdy, sdz = (float(X.std()), float(Y.std()), float(Z.std()))

    # density per panel (in panel-local svg coords) for accumulation shading ---
    def panel_density(ax, ay):
        P = np.column_stack([np.asarray([sx(v) for v in ax]),
                             np.asarray([sy(v) for v in ay])])
        # _local_density expects coords inside [0,w]; feed panel width PW
        return P, _local_density(P, PW, H, gx=40, gy=30)

    panels = [
        ("XY · top-down", X, Y, "x", "y"),
        ("XZ · front", X, Z, "x", "z"),
        ("YZ · side", Y, Z, "y", "z"),
    ]

    # majors at multiples of 0.2, minors at 0.1 -----------------------------
    majors = np.round(np.arange(-SCALE, SCALE + 1e-9, 0.2), 2)
    minors = np.round(np.arange(-SCALE, SCALE + 1e-9, 0.1), 2)

    body = []
    for pi, (title, ax, ay, axl, ayl) in enumerate(panels):
        ox = pi * PW
        g = [f'<g transform="translate({ox},0)">']
        g.append(f'<text x="{bL}" y="20" font-size="9.5" letter-spacing=".10em" '
                 f'fill="var(--ink-faint)">{title}</text>')

        # --- fine numeric scale: minors (hairline) then majors (labelled) ---
        gframe = ['<g class="ax">']
        for t in minors:
            xx, yy = sx(t), sy(t)
            gframe.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bot}" '
                          f'stroke="var(--rule-soft)" stroke-width="0.4" stroke-opacity="0.55"/>')
            gframe.append(f'<line x1="{bL}" y1="{yy:.1f}" x2="{bR}" y2="{yy:.1f}" '
                          f'stroke="var(--rule-soft)" stroke-width="0.4" stroke-opacity="0.55"/>')
        for t in majors:
            xx, yy = sx(t), sy(t)
            gframe.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bot}" '
                          f'stroke="var(--rule)" stroke-width="0.7"/>')
            gframe.append(f'<line x1="{bL}" y1="{yy:.1f}" x2="{bR}" y2="{yy:.1f}" '
                          f'stroke="var(--rule)" stroke-width="0.7"/>')
            # bottom-edge x labels + left-edge y labels (zero emphasised)
            zero = abs(t) < 1e-9
            tv = 0.0 if zero else t            # avoid -0.0 on the centre tick
            lab = "0.0" if zero else f"{tv:+.1f}"
            lc = "var(--ink-soft)" if zero else "var(--ink-faint)"
            gframe.append(f'<text x="{xx:.1f}" y="{bot+12}" font-size="8" '
                          f'fill="{lc}" text-anchor="middle" '
                          f'font-family="ui-monospace,monospace">{lab}</text>')
            gframe.append(f'<text x="{bL-4:.1f}" y="{yy+3:.1f}" font-size="8" '
                          f'fill="{lc}" text-anchor="end" '
                          f'font-family="ui-monospace,monospace">{lab}</text>')
        # box outline
        gframe.append(f'<rect x="{bL}" y="{top}" width="{bR-bL:.1f}" height="{bot-top:.1f}" '
                      f'fill="none" stroke="var(--rule)" stroke-width="1"/>')
        # axis letters
        gframe.append(f'<text x="{bR}" y="{bot+12}" font-size="8.5" fill="var(--ink-faint)" '
                      f'text-anchor="end" font-style="italic">{axl}</text>')
        gframe.append(f'<text x="{bL-4:.1f}" y="{top+9}" font-size="8.5" fill="var(--ink-faint)" '
                      f'text-anchor="end" font-style="italic">{ayl}</text>')
        gframe.append('</g>')
        g.extend(gframe)

        # --- accumulation dots: faint ink, accent only on the dense core ----
        P, dens = panel_density(ax, ay)
        hot_th = np.quantile(dens, 0.96)
        core_th = np.quantile(dens, 0.995)
        for i in range(len(P)):
            px, py = P[i, 0], P[i, 1]   # P already panel-local
            d = dens[i]
            if d >= core_th:
                g.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.0" '
                         f'fill="var(--accent)" fill-opacity="0.95"/>')
            elif d >= hot_th:
                # warm accumulation ridge toward the core, still restrained
                mix = min(1.0, (d - hot_th) / max(core_th - hot_th, 1e-9))
                op = 0.45 + 0.35 * mix
                g.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.5" '
                         f'fill="var(--accent)" fill-opacity="{op:.2f}"/>')
            else:
                g.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.1" '
                         f'fill="var(--ink-faint)" fill-opacity="0.34"/>')

        # --- single accent core marker (centroid of densest cells) ----------
        core_mask = dens >= core_th
        if core_mask.any():
            cx = float(P[core_mask, 0].mean())
            cy = float(P[core_mask, 1].mean())
            g.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.6" fill="none" '
                     f'stroke="var(--accent)" stroke-width="1.4" stroke-opacity="0.9"/>')

        # --- extent caliper: brackets the data span on each axis, so the
        #     reader can read the Z-thinness straight off the shared scale ----
        lo_a, hi_a = float(ax.min()), float(ax.max())
        lo_b, hi_b = float(ay.min()), float(ay.max())
        yb = bot + 19
        if True:
            g.append(f'<line x1="{sx(lo_a):.1f}" y1="{yb}" x2="{sx(hi_a):.1f}" y2="{yb}" '
                     f'stroke="var(--good)" stroke-width="1.4"/>')
            g.append(f'<line x1="{sx(lo_a):.1f}" y1="{yb-2.5}" x2="{sx(lo_a):.1f}" y2="{yb+2.5}" stroke="var(--good)" stroke-width="1.4"/>')
            g.append(f'<line x1="{sx(hi_a):.1f}" y1="{yb-2.5}" x2="{sx(hi_a):.1f}" y2="{yb+2.5}" stroke="var(--good)" stroke-width="1.4"/>')
            xb = bL - 16
            g.append(f'<line x1="{xb}" y1="{sy(lo_b):.1f}" x2="{xb}" y2="{sy(hi_b):.1f}" '
                     f'stroke="var(--good)" stroke-width="1.4"/>')
            g.append(f'<line x1="{xb-2.5}" y1="{sy(lo_b):.1f}" x2="{xb+2.5}" y2="{sy(lo_b):.1f}" stroke="var(--good)" stroke-width="1.4"/>')
            g.append(f'<line x1="{xb-2.5}" y1="{sy(hi_b):.1f}" x2="{xb+2.5}" y2="{sy(hi_b):.1f}" stroke="var(--good)" stroke-width="1.4"/>')

        g.append('</g>')
        body.append("".join(g))

    aria = ("Orthographic triptych of the 3-D embedding reservoir: XY top-down, "
            "XZ front, YZ side, all at one identical 0.2-major numeric scale. The "
            "cloud fills X and Y but is visibly compressed in Z, so the thin-in-Z "
            "anisotropy is measurable across panels. Faint ink dots accumulate into "
            "density; the dense megacluster core carries the single accent in every "
            "panel; green calipers bracket the data extent on each axis.")

    thin = min(sdx, sdy, sdz)
    broad = max(sdx, sdy, sdz)
    ratio = broad / max(thin, 1e-9)

    return {
        "num": "3D 02", "order": 21, "name": "Orthographic triptych",
        "tech": "xy · xz · yz",
        "why": (f"The same 3-D reservoir under three undistorted axis-aligned "
                f"projections at one shared scale (0.2 majors, 0.1 minors). σ is "
                f"{sdx:.2f}/{sdy:.2f}/{sdz:.2f} along x/y/z — the cloud is ~{ratio:.1f}× "
                f"broader on its widest axis than its thinnest, and that flattening "
                f"is legible as a shorter caliper in the panels that include z."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="a"></i> megacluster core (shared accent across panels)</span>'
                  '<span><i class="f"></i> cloud point — density by accumulation</span>'
                  '<span><i class="g"></i> data-extent caliper (more space = higher resolution)</span>',
        "reveal": (f"<b>Reveals:</b> occupancy <b>anisotropy</b> — at one shared "
                   f"0.2-major scale the cloud spans the full box in x/y but is "
                   f"<b>thin in z</b> (σ {sdz:.2f} vs {max(sdx,sdy):.2f}). A 2-D "
                   f"footprint hides this; the matched ortho scale makes the "
                   f"flattening directly measurable across planes."),
        "cls": "",
    }
