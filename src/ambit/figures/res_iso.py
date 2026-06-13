from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_res_iso(ctx):
    # ---- score: fraction of nominal dims the variance actually occupies ----
    eigs = np.asarray(ctx.eigs, float)
    dim = int(ctx.scan.dim)
    erank = metrics.effective_rank(eigs)          # continuous effective dimensionality
    pratio = metrics.participation_ratio(eigs)    # second-moment occupancy
    score = float(np.clip(erank / max(1, dim), 0.0, 1.0))   # 0..1 utilization
    pr_score = float(np.clip(pratio / max(1, dim), 0.0, 1.0))

    # ---- gauge geometry (own layout, template-matched proportions) ----
    W, H = 620, 170
    x0, x1 = 70.0, 560.0           # track left/right (0 .. 1)
    span = x1 - x0
    ty = 95.0                       # track baseline
    band_y, band_h = 100.0, 15.0

    def X(v):
        return x0 + float(np.clip(v, 0.0, 1.0)) * span

    mx = X(score)                   # dataset marker x

    body = []

    # large dataset readout (the single accent emphasis), above the marker
    rx = float(np.clip(mx, x0 + 56, x1 - 56))   # keep label on-canvas
    body.append(
        f'<text x="{rx:.1f}" y="34" fill="var(--accent)" font-size="22" font-weight="700" '
        f'text-anchor="middle" style="font-variant-numeric:tabular-nums">isoscore = {score:.2f}</text>'
    )
    body.append(
        f'<text x="{rx:.1f}" y="50" fill="var(--ink-faint)" font-size="9.5" text-anchor="middle">'
        f'space utilization · 0..1</text>'
    )

    # graded utilization band: collapsed(bad, left) -> isotropic(good, right).
    # discrete color-mix steps (a continuous gauge band, not a scatter) so we avoid
    # a url(#id) gradient reference; a density-style bad->good ramp is legit here.
    nseg = 60
    seg_w = span / nseg
    for s in range(nseg):
        t = (s + 0.5) / nseg                 # 0..1 across the band
        if t <= 0.5:                         # collapsed half: bad fading out
            f = (0.5 - t) / 0.5              # 1 at left edge -> 0 at midpoint
            pct = 8 + 47 * f
            col = f"color-mix(in srgb, var(--bad) {pct:.0f}%, var(--panel))"
        else:                                # isotropic half: good fading in
            f = (t - 0.5) / 0.5              # 0 at midpoint -> 1 at right edge
            pct = 8 + 47 * f
            col = f"color-mix(in srgb, var(--good) {pct:.0f}%, var(--panel))"
        sx = x0 + s * seg_w
        body.append(
            f'<rect x="{sx:.2f}" y="{band_y:.1f}" width="{seg_w + 0.6:.2f}" '
            f'height="{band_h:.1f}" fill="{col}" stroke="none"/>'
        )
    # thin track baseline
    body.append(
        f'<line x1="{x0:.1f}" y1="{ty:.1f}" x2="{x1:.1f}" y2="{ty:.1f}" '
        f'stroke="var(--rule-soft)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
    )

    # minor ticks every 0.05 (skip the quarter marks, those are major)
    minor = []
    for k in range(1, 20):
        v = k * 0.05
        if abs((v * 4) - round(v * 4)) < 1e-9:   # 0.25/0.5/0.75 -> major
            continue
        minor.append(v)
    body.append('<g stroke="var(--rule-soft)" vector-effect="non-scaling-stroke">')
    for v in minor:
        xv = X(v)
        body.append(f'<line x1="{xv:.1f}" y1="92" x2="{xv:.1f}" y2="{ty:.1f}"/>')
    body.append('</g>')
    body.append('<g fill="var(--ink-faint)" font-size="7" text-anchor="middle" '
                'style="font-variant-numeric:tabular-nums">')
    for v in minor:
        xv = X(v)
        body.append(f'<text x="{xv:.1f}" y="88">.{int(round(v*100)):02d}</text>')
    body.append('</g>')

    # major ticks 0, 0.25, 0.5, 0.75, 1.0
    for v, lab in [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")]:
        xv = X(v)
        body.append(
            f'<line x1="{xv:.1f}" y1="90" x2="{xv:.1f}" y2="{ty:.1f}" '
            f'stroke="var(--ink-faint)" vector-effect="non-scaling-stroke"/>'
            f'<text x="{xv:.1f}" y="82" fill="var(--ink-faint)" font-size="10" text-anchor="middle" '
            f'style="font-variant-numeric:tabular-nums">{lab}</text>'
        )

    # dashed faint reference: perfect isotropy = 1.00
    body.append(
        f'<line x1="{x1:.1f}" y1="60" x2="{x1:.1f}" y2="126" stroke="var(--ink-faint)" '
        f'stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>'
        f'<text x="{x1-4:.1f}" y="58" fill="var(--ink-faint)" font-size="9.5" text-anchor="end" '
        f'style="font-variant-numeric:tabular-nums">ideal isotropy = 1.00</text>'
    )

    # secondary faint reference: participation-ratio occupancy (a stricter read)
    px = X(pr_score)
    body.append(
        f'<line x1="{px:.1f}" y1="98" x2="{px:.1f}" y2="117" stroke="var(--ink-faint)" '
        f'stroke-dasharray="2 2" vector-effect="non-scaling-stroke"/>'
        f'<text x="{px:.1f}" y="166" fill="var(--ink-faint)" font-size="8" text-anchor="middle" '
        f'style="font-variant-numeric:tabular-nums">PR {pr_score:.2f}</text>'
    )

    # dataset marker: the one accent emphasis
    body.append(
        f'<line x1="{mx:.1f}" y1="60" x2="{mx:.1f}" y2="122" stroke="var(--accent)" '
        f'stroke-width="2.2" vector-effect="non-scaling-stroke"/>'
        f'<circle cx="{mx:.1f}" cy="{band_y + band_h/2:.1f}" r="3.5" fill="var(--accent)" '
        f'vector-effect="non-scaling-stroke"/>'
    )

    # endpoint + directional labels under the band
    body.append(
        f'<text x="{x0:.1f}" y="134" fill="var(--ink-faint)" font-size="10" text-anchor="start">collapsed</text>'
        f'<text x="{x1:.1f}" y="134" fill="var(--ink-faint)" font-size="10" text-anchor="end">isotropic</text>'
        f'<text x="{x0:.1f}" y="150" fill="var(--ink-faint)" font-size="8.5" text-anchor="start">'
        f'all variance on one axis</text>'
        f'<text x="{x1:.1f}" y="150" fill="var(--ink-faint)" font-size="8.5" text-anchor="end">'
        f'variance equal across the sphere</text>'
    )

    aria = (f"Space-utilization gauge: a horizontal 0-to-1 track with major ticks at 0, 0.25, 0.5, "
            f"0.75 and 1.0 and minor ticks every 0.05, beneath a graded band running from collapsed "
            f"(bad) on the left to isotropic (good) on the right. The dataset marker is an accent tick "
            f"and dot at isoscore {score:.2f} (effective rank {erank:.1f} of {dim} dimensions), with a "
            f"dashed reference at 1.0 for perfect isotropy and a faint participation-ratio mark at "
            f"{pr_score:.2f}.")

    return {
        "num": "RES 03", "order": 92, "name": "Space-utilization gauge", "tech": "isoscore",
        "why": (f"Effective rank {erank:.1f} of {dim} nominal dims = a {score:.2f} utilization score. "
                f"Far left means variance has collapsed onto a few axes; the right edge is a fully "
                f"isotropic space using every dimension equally."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": ('<span><i class="a"></i> dataset isoscore</span>'
                   '<span><i class="dash"></i> ideal isotropy = 1.0</span>'
                   '<span><i class="f"></i> collapsed→isotropic band</span>'),
        "reveal": (f"<b>Reveals:</b> how much of the embedding space the dataset actually uses — "
                   f"here only ≈{score*100:.0f}% ({erank:.0f} of {dim} effective dims), so most "
                   f"of the nominal capacity is unused."),
        "cls": "",
    }
