from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


def _kde(samples, grid, bw):
    """Tiny gaussian KDE -> density over `grid` (no scipy needed)."""
    s = np.asarray(samples, float)
    if s.size == 0:
        return np.zeros_like(grid)
    d = (grid[:, None] - s[None, :]) / bw
    k = np.exp(-0.5 * d * d)
    return k.sum(1) / (s.size * bw * np.sqrt(2 * np.pi))


@figure
def fig_res_wb(ctx):
    W, H = 760, 470
    L, R, T, B = 96, 712, 60, 392          # plot box in svg coords
    AX_LO, AX_HI = -1.0, 1.0               # cosine axis range

    def X(v):
        v = np.asarray(v, float)
        return L + (v - AX_LO) / (AX_HI - AX_LO) * (R - L)

    body = []

    # ---- header -----------------------------------------------------------------
    body.append(f'<text x="{L}" y="36" font-size="13" font-weight="700" fill="var(--ink)">'
                f'within- vs between-cluster cosine</text>')

    labels = ctx.labels
    X64 = np.asarray(ctx.es.X, float)      # (m, d) L2-normalized reservoir

    # ---- degrade gracefully if no cluster labels --------------------------------
    if labels is None or len(np.unique(np.asarray(labels))) < 2:
        P = _box(ctx.xy, W, H)
        for i in range(len(P)):
            body.append(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.2" '
                        f'fill="var(--ink-faint)" fill-opacity="0.4"/>')
        body.append(f'<text x="{W/2:.0f}" y="{H/2:.0f}" text-anchor="middle" font-size="12" '
                    f'fill="var(--ink-faint)">needs cluster labels</text>')
        aria = ("Within- versus between-cluster cosine separability is unavailable because no "
                "cluster labels are present; the faint projected reservoir cloud is shown instead.")
        return {"num": "RES 05", "order": 94, "name": "Within- vs between-cluster cosine",
                "tech": "separability",
                "why": ("Compares the cosine of same-cluster pairs against different-cluster pairs; "
                        "a clean rightward gap between the two means clusters are separable."),
                "svg": _svg(W, H, aria, "".join(body)),
                "legend": '<span><i class="f"></i> needs cluster labels</span>',
                "reveal": "<b>Reveals:</b> needs cluster labels.", "cls": ""}

    lab = np.asarray(labels)
    uniq = np.unique(lab)
    code = np.searchsorted(uniq, lab)      # 0..K-1 integer codes
    m = len(code)

    # ensure unit-norm rows so dot products are true cosines in [-1, 1]
    nrm = np.linalg.norm(X64, axis=1, keepdims=True)
    X64 = X64 / np.where(nrm > 0, nrm, 1.0)

    # ---- sample within- and between-cluster pairs -------------------------------
    rng = np.random.default_rng(0)
    n_pairs = 60000
    a = rng.integers(0, m, n_pairs)
    b = rng.integers(0, m, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    cos_all = np.einsum("ij,ij->i", X64[a], X64[b])   # already unit-norm rows
    same = code[a] == code[b]
    within = cos_all[same]
    between = cos_all[~same]
    # cap each to a balanced, bounded sample for stable density
    if within.size > 30000:
        within = rng.choice(within, 30000, replace=False)
    if between.size > 30000:
        between = rng.choice(between, 30000, replace=False)

    w_mean, w_sd = float(within.mean()), float(within.std())
    b_mean, b_sd = float(between.mean()), float(between.std())
    separation = w_mean - b_mean                       # within sits higher when >0

    # ---- density grids over the shared cosine axis ------------------------------
    grid = np.linspace(AX_LO, AX_HI, 320)
    bw = max(0.02, 0.9 * min(w_sd, b_sd) * (within.size + between.size) ** (-0.2))
    dens_w = _kde(within, grid, bw)
    dens_b = _kde(between, grid, bw)
    dmax = max(dens_w.max(), dens_b.max(), 1e-9)

    def Y(d):
        return B - (np.asarray(d, float) / dmax) * (B - T) * 0.92

    # ---- minor-tick ruler: hairline gridlines + ticks + mono labels at 0.1 ------
    minor = np.round(np.arange(-0.9, 1.0, 0.1), 1)
    major = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    minor = np.array([t for t in minor if not np.any(np.isclose(t, major))])
    body.append('<g aria-hidden="true">')
    for t in minor:
        xx = float(X(t))
        body.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{B}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.6"/>')
    body.append('</g>')

    # ---- overlap (confusable) zone: shade min(dens_w, dens_b) -------------------
    ov = np.minimum(dens_w, dens_b)
    xs = X(grid)
    pts = [f"{xs[0]:.1f},{B}"]
    pts += [f"{xs[i]:.1f},{Y(ov[i]):.1f}" for i in range(len(grid))]
    pts.append(f"{xs[-1]:.1f},{B}")
    body.append(f'<path d="M {pts[0]} L ' + " L ".join(pts[1:]) +
                f' Z" fill="color-mix(in srgb, var(--bad) 16%, transparent)" stroke="none"/>')

    # ---- baseline rule ----------------------------------------------------------
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')

    # ---- major ticks + labels + minor ticks + minor mono labels -----------------
    for t in major:
        xx = float(X(t))
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+6}" '
                    f'stroke="var(--ink-faint)" stroke-width="0.8"/>')
        lbl = f"{t:.1f}".replace("-", "−")
        body.append(f'<text x="{xx:.1f}" y="{B+20}" text-anchor="middle" font-size="11" '
                    f'fill="var(--ink-faint)">{lbl}</text>')
    body.append('<g aria-hidden="true">')
    for t in minor:
        xx = float(X(t))
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+4}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.6"/>')
    body.append('</g>')
    body.append('<g aria-hidden="true">')
    for t in minor:
        xx = float(X(t))
        lbl = f"{t:.1f}".replace("-", "−")
        body.append(f'<text x="{xx:.1f}" y="{B+20}" text-anchor="middle" font-size="8.5" '
                    f'fill="var(--ink-faint)" font-family="ui-monospace,monospace" '
                    f'font-variant-numeric="tabular-nums" opacity="0.65">{lbl}</text>')
    body.append('</g>')

    # ---- axis titles ------------------------------------------------------------
    body.append(f'<text x="{(L+R)/2:.0f}" y="432" text-anchor="middle" font-size="11" '
                f'fill="var(--ink-soft)" letter-spacing="0.04em">cosine similarity</text>')
    body.append(f'<text x="86" y="{(T+B)/2:.0f}" text-anchor="middle" font-size="9.5" '
                f'fill="var(--ink-faint)" transform="rotate(-90 86 {(T+B)/2:.0f})">density</text>')

    # ---- ideal isotropic between-peak reference (cosine 0) ----------------------
    x0 = float(X(0.0))
    body.append(f'<line x1="{x0:.1f}" y1="{B}" x2="{x0:.1f}" y2="150" '
                f'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3"/>')
    body.append(f'<text x="{x0-9:.1f}" y="186" text-anchor="end" font-size="10.5" '
                f'fill="var(--ink-faint)">ideal isotropic</text>')
    body.append(f'<text x="{x0-9:.1f}" y="199" text-anchor="end" font-size="10.5" '
                f'fill="var(--ink-faint)">between-peak = 0</text>')

    # ---- between-cluster density curve (the side that should sit at 0) ----------
    bcol = "color-mix(in srgb, var(--bad) 70%, var(--ink-soft))"
    bpath = " ".join(f"{('M' if i==0 else 'L')} {xs[i]:.1f},{Y(dens_b[i]):.1f}"
                     for i in range(len(grid)))
    body.append(f'<path d="{bpath}" fill="none" stroke="{bcol}" stroke-width="1.4" '
                f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')

    # ---- within-cluster density curve (better-separated reading = good) ---------
    wpath = " ".join(f"{('M' if i==0 else 'L')} {xs[i]:.1f},{Y(dens_w[i]):.1f}"
                     for i in range(len(grid)))
    body.append(f'<path d="{wpath}" fill="none" stroke="var(--good)" stroke-width="1.4" '
                f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')

    # ---- peak markers + leader labels -------------------------------------------
    bpk_x = float(X(grid[int(np.argmax(dens_b))]))
    bpk_y = float(Y(dens_b.max()))
    wpk_x = float(X(grid[int(np.argmax(dens_w))]))
    wpk_y = float(Y(dens_w.max()))

    body.append(f'<circle cx="{bpk_x:.1f}" cy="{bpk_y:.1f}" r="3" fill="{bcol}"/>')
    body.append(f'<line x1="{bpk_x:.1f}" y1="{bpk_y:.1f}" x2="{bpk_x:.1f}" y2="44" '
                f'stroke="{bcol}" stroke-width="0.7" stroke-dasharray="2 3" opacity="0.7"/>')
    body.append(f'<text x="{bpk_x-8:.1f}" y="{bpk_y-16:.1f}" text-anchor="end" font-size="10.5" '
                f'fill="{bcol}">between-cluster</text>')
    body.append(f'<text x="{bpk_x-8:.1f}" y="{bpk_y-3:.1f}" text-anchor="end" font-size="10.5" '
                f'fill="var(--ink-faint)">peak ≈ {b_mean:.2f} · sd {b_sd:.2f}</text>')

    body.append(f'<circle cx="{wpk_x:.1f}" cy="{wpk_y:.1f}" r="3" fill="var(--good)"/>')
    body.append(f'<line x1="{wpk_x:.1f}" y1="{wpk_y:.1f}" x2="{wpk_x:.1f}" y2="44" '
                f'stroke="var(--good)" stroke-width="0.7" stroke-dasharray="2 3" opacity="0.7"/>')
    # flip the within label to the left when its peak hugs the right edge (high cosine = well separated)
    if wpk_x > R - 120:
        wlx, wanchor = wpk_x - 8, "end"
    else:
        wlx, wanchor = wpk_x + 8, "start"
    body.append(f'<text x="{wlx:.1f}" y="{wpk_y+2:.1f}" text-anchor="{wanchor}" font-size="10.5" '
                f'fill="var(--good)">within-cluster</text>')
    body.append(f'<text x="{wlx:.1f}" y="{wpk_y+15:.1f}" text-anchor="{wanchor}" font-size="10.5" '
                f'fill="var(--ink-faint)">peak ≈ {w_mean:.2f} · sd {w_sd:.2f}</text>')

    # ---- separation caliper between the two peaks (accent) ----------------------
    cl, cr = (bpk_x, wpk_x) if bpk_x <= wpk_x else (wpk_x, bpk_x)
    body.append(f'<line x1="{cl:.1f}" y1="44" x2="{cr:.1f}" y2="44" stroke="var(--accent)" stroke-width="2.0"/>')
    body.append(f'<line x1="{cl:.1f}" y1="39" x2="{cl:.1f}" y2="49" stroke="var(--accent)" stroke-width="2.0"/>')
    body.append(f'<line x1="{cr:.1f}" y1="39" x2="{cr:.1f}" y2="49" stroke="var(--accent)" stroke-width="2.0"/>')
    body.append(f'<text x="{(cl+cr)/2:.1f}" y="35" text-anchor="middle" font-size="11" font-weight="700" '
                f'fill="var(--accent)" font-variant-numeric="tabular-nums">separation = {separation:+.2f}</text>')

    # ---- confusable-zone caption (peak of the overlap) --------------------------
    oi = int(np.argmax(ov))
    ozx = float(xs[oi])
    body.append(f'<text x="{ozx:.1f}" y="312" text-anchor="middle" font-size="10.5" '
                f'fill="var(--bad)">confusable zone</text>')
    body.append(f'<text x="{ozx:.1f}" y="325" text-anchor="middle" font-size="9.5" '
                f'fill="var(--ink-faint)">same vs different unresolvable</text>')

    # ---- subtitle ---------------------------------------------------------------
    body.append(f'<text x="{L}" y="52" font-size="10.5" fill="var(--ink-faint)" letter-spacing="0.03em">'
                f'{len(uniq)} clusters · {within.size + between.size} sampled pairs · separability</text>')

    sep_tone = "var(--good)" if separation > 0 else "var(--bad)"
    verdict = "separable" if separation > 0 else "entangled"
    aria = (f"Within- versus between-cluster cosine similarity on a shared cosine axis from -1 to 1, "
            f"over {len(uniq)} clusters. The within-cluster density (same-cluster pairs) peaks at "
            f"cosine {w_mean:.2f} with sd {w_sd:.2f}; the between-cluster density (different-cluster "
            f"pairs) peaks at cosine {b_mean:.2f} with sd {b_sd:.2f}, relative to the ideal isotropic "
            f"between-peak at cosine 0 marked by a dashed reference rule. The two densities overlap in "
            f"a shaded confusable zone where same and different are unresolvable; an accent caliper "
            f"spans the gap between the two peaks for a separation margin of {separation:+.2f}, so the "
            f"clusters read as {verdict}. A full 0.1-step minor-tick ruler runs along the cosine axis.")

    legend = ('<span><i class="g"></i> within-cluster (same)</span>'
              '<span><i class="b"></i> between-cluster (different)</span>'
              '<span><i class="f"></i> confusable overlap · isotropic ref</span>')

    reveal = ("<b>Reveals:</b> cluster separability in the native cosine geometry — a within-cluster "
              "hump sitting cleanly to the right of the between-cluster hump (large positive separation) "
              "means same-cluster items are reliably more similar than cross-cluster items; heavy overlap "
              "means the clustering is not resolved by cosine alone.")

    return {"num": "RES 05", "order": 94, "name": "Within- vs between-cluster cosine",
            "tech": "separability",
            "why": ("Samples same-cluster and different-cluster pairs and draws their cosine distributions "
                    "on a shared axis: the within-cluster hump should sit to the right of the between-cluster "
                    "hump, and the rightward separation margin measures how cleanly cosine resolves the clusters."),
            "svg": _svg(W, H, aria, "".join(body)),
            "legend": legend, "reveal": reveal, "cls": ""}
