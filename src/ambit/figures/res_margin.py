from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


def _nice_step(span, target=10):
    """A 1/2/5 * 10^k step that yields ~target ticks across span."""
    if span <= 0:
        return 1.0
    raw = span / target
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def _fmt(v):
    """Compact tick label, no leading zero, trailing zeros trimmed."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    if s in ("", "-", "."):
        s = "0"
    return s


@figure
def fig_res_margin(ctx):
    w, h = 760, 470
    # plot box
    L, R, T, B = 92, 732, 88, 404

    # ---- degrade gracefully if the kNN backend produced no distances
    if ctx.knn_dist is None:
        P = _box(ctx.xy, w, h)
        dots = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.2" '
            f'fill="var(--ink-faint)" fill-opacity="0.35"/>'
            for x, y in P
        )
        note = ('<text x="380" y="235" font-size="12" fill="var(--ink-faint)" '
                'text-anchor="middle">needs kNN backend (ctx.knn_dist)</text>')
        body = dots + note
        return {"num": "RES 04", "order": 93,
                "name": "Nearest-neighbor cosine margin", "tech": "top1 vs top2",
                "why": "Per-item margin between the top-1 and top-2 cosine "
                       "neighbor; larger margins mean the best match is "
                       "decisively separated from the runner-up.",
                "svg": _svg(w, h, "Nearest-neighbor cosine margin unavailable: "
                            "no kNN distances in reservoir.", body),
                "legend": '<span class="leg">needs kNN backend</span>',
                "reveal": "<b>Reveals:</b> retrieval decisiveness once a "
                          "neighbor index is available.",
                "cls": ""}

    # ---- margin = sim(top1) - sim(top2), sim = 1 - dist
    sim = 1.0 - np.asarray(ctx.knn_dist, float)
    if sim.shape[1] < 2:
        s = np.sort(sim, axis=1)
        margin = np.zeros(sim.shape[0])
    else:
        s = np.sort(sim, axis=1)[:, ::-1]
        margin = np.clip(s[:, 0] - s[:, 1], 0.0, None)
    n = margin.shape[0]
    med = float(np.median(margin))

    # ---- isotropic reference: under perfect isotropy in `dim` dims, the
    # expected cos with the nearest of m random directions ~ small; the
    # top1/top2 gap of an isotropic cloud is a useful "no-structure" yardstick.
    dim = int(getattr(ctx.scan, "dim", sim.shape[1]) or sim.shape[1])
    rng = np.random.default_rng(0)
    m_ref = min(n, 1500)
    k_ref = sim.shape[1]
    ref = rng.standard_normal((m_ref, dim))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
    q = rng.standard_normal((m_ref, dim))
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    # cosine of each query to a small random panel, take top-k gap
    panel = rng.standard_normal((max(k_ref + 2, 12), dim))
    panel /= np.linalg.norm(panel, axis=1, keepdims=True) + 1e-12
    cref = q @ panel.T
    cref = np.sort(cref, axis=1)[:, ::-1]
    ref_margin = float(np.median(cref[:, 0] - cref[:, 1]))

    # ---- axis: 0 .. nice upper bound covering the bulk (q99, capped at max)
    hi_data = float(max(np.quantile(margin, 0.99), med * 3.0, ref_margin * 1.15,
                        np.max(margin) * 0.6))
    hi_data = min(hi_data, float(np.max(margin)))
    step = _nice_step(hi_data, target=10)
    n_steps = int(np.ceil(hi_data / step + 1e-9))
    xmax = step * n_steps
    if xmax <= 0:
        xmax = step

    # ---- histogram with fine bins, clip overflow into last bin
    nb = 40
    edges = np.linspace(0.0, xmax, nb + 1)
    mc = np.clip(margin, 0.0, xmax - 1e-12)
    counts, _ = np.histogram(mc, bins=edges)
    cmax = max(int(counts.max()), 1)

    def X(v):
        return L + (v / xmax) * (R - L)

    def Y(c):
        return B - (c / cmax) * (B - T)

    # near-tie (bad) vs decisive (good) split at the median margin
    split = med
    bars = []
    for i in range(nb):
        c = int(counts[i])
        if c == 0:
            continue
        x0 = X(edges[i]); x1 = X(edges[i + 1])
        y = Y(c)
        center = 0.5 * (edges[i] + edges[i + 1])
        tok = "--bad" if center <= split else "--good"
        bars.append(
            f'<rect x="{x0 + 0.6:.1f}" y="{y:.1f}" '
            f'width="{(x1 - x0) - 1.2:.1f}" height="{B - y:.1f}" '
            f'fill="color-mix(in srgb, var({tok}) 55%, var(--panel))"/>'
        )
    bars_svg = "<g>" + "".join(bars) + "</g>"

    # ---- gridlines + ticks at every nice step
    ticks = [step * i for i in range(n_steps + 1)]
    grid = "".join(
        f'<line x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{B}"/>'
        for t in ticks if 0 < t < xmax + 1e-9
    )
    grid_svg = f'<g stroke="var(--rule-soft)" stroke-width="0.6">{grid}</g>'

    minor = []
    for i in range(n_steps):
        midv = step * (i + 0.5)
        if midv < xmax:
            minor.append(f'<line x1="{X(midv):.1f}" y1="{B}" '
                         f'x2="{X(midv):.1f}" y2="{B + 4}"/>')
    minor_svg = f'<g stroke="var(--ink-faint)" stroke-width="0.7">{"".join(minor)}</g>'

    major = "".join(
        f'<line x1="{X(t):.1f}" y1="{B}" x2="{X(t):.1f}" y2="{B + 7}"/>'
        for t in ticks
    )
    major_svg = f'<g stroke="var(--ink-faint)" stroke-width="0.9">{major}</g>'

    tlabels = "".join(
        f'<text x="{X(t):.1f}" y="{B + 19}">{_fmt(t)}</text>' for t in ticks
    )
    tlabels_svg = (
        '<g font-family="ui-monospace,\'SF Mono\',Menlo,Consolas,monospace" '
        'font-size="9" fill="var(--ink-soft)" text-anchor="middle" '
        'style="font-variant-numeric:tabular-nums">' + tlabels + "</g>"
    )

    # ---- y-axis: a couple of count gridmarks (left labels)
    yticks = [0, int(round(cmax * 0.5)), cmax]
    ylabels = "".join(
        f'<text x="{L - 8}" y="{Y(c) + 3:.1f}" font-size="9" '
        f'fill="var(--ink-faint)" text-anchor="end">{c:,}</text>'
        for c in sorted(set(yticks))
    )

    baseline = (f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" '
                f'stroke="var(--rule)" stroke-width="1"/>')

    # ---- markers: isotropic reference median (dashed) + dataset median (accent)
    ref_x = X(min(ref_margin, xmax))
    med_x = X(min(med, xmax))
    ref_in = ref_margin <= xmax
    ref_marker = ""
    if ref_in:
        ref_marker = (
            f'<line x1="{ref_x:.1f}" y1="{T}" x2="{ref_x:.1f}" y2="{B}" '
            f'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<text x="{ref_x + 5:.1f}" y="{T - 8}" font-size="9.5" '
            f'fill="var(--ink-faint)" text-anchor="start">'
            f'isotropic ref median = {_fmt(ref_margin)}</text>'
        )
    # keep the accent median label inside the box
    med_anchor = "start" if med_x < R - 150 else "end"
    med_dx = 6 if med_anchor == "start" else -6
    med_marker = (
        f'<line x1="{med_x:.1f}" y1="{T}" x2="{med_x:.1f}" y2="{B}" '
        f'stroke="var(--accent)" stroke-width="2.2"/>'
        f'<circle cx="{med_x:.1f}" cy="{T}" r="3" fill="var(--accent)"/>'
        f'<text x="{med_x + med_dx:.1f}" y="{T - 8}" font-size="10.5" '
        f'font-weight="700" fill="var(--accent)" text-anchor="{med_anchor}">'
        f'median margin = {_fmt(med)}</text>'
    )

    # ---- titles & directional annotations
    title = (
        f'<text x="{L}" y="24" font-size="13" font-weight="700" '
        f'fill="var(--ink)">per-item NN margin  =  cos(top-1) − cos(top-2)</text>'
        f'<text x="{R}" y="24" font-size="10.5" fill="var(--ink-faint)" '
        f'text-anchor="end">{n:,} items</text>'
        f'<text x="{L}" y="40" font-size="10.5" fill="var(--ink-faint)">'
        f'angular resolution at the retrieval boundary — larger margin = '
        f'more decisively resolved</text>'
    )

    floor_note = (f'<text x="{L}" y="{B + 8}" font-size="8.5" fill="var(--bad)" '
                  f'text-anchor="start">resolution floor (margin→0, near-tie)</text>')
    xtitle = (f'<text x="{R}" y="{B + 34}" font-size="9" fill="var(--ink-faint)" '
              f'text-anchor="end">margin  =  cos(top-1) − cos(top-2)</text>')
    ytitle = (f'<text x="70" y="{(T + B) / 2:.1f}" font-size="8.5" '
              f'fill="var(--ink-faint)" text-anchor="middle" '
              f'transform="rotate(-90 70 {(T + B) / 2:.1f})">'
              f'items per {_fmt(xmax / nb)} margin bin</text>')

    bad_anno = (f'<text x="{X(split * 0.5):.1f}" y="{T - 12}" font-size="9.5" '
                f'fill="var(--bad)" text-anchor="middle">near-tie crowding</text>')
    good_cx = X(min(xmax * 0.78, max(split * 2.5, xmax * 0.6)))
    good_anno = (
        f'<text x="{good_cx:.1f}" y="{T + 70}" font-size="9.5" '
        f'fill="var(--good)" text-anchor="middle">decisively resolved</text>'
        f'<text x="{good_cx:.1f}" y="{T + 82}" font-size="9.5" '
        f'fill="var(--good)" text-anchor="middle">tail — large margin,</text>'
        f'<text x="{good_cx:.1f}" y="{T + 94}" font-size="9.5" '
        f'fill="var(--good)" text-anchor="middle">high resolution</text>'
    )

    body = (
        title + grid_svg + bars_svg + baseline + minor_svg + major_svg +
        tlabels_svg + ylabels + floor_note + xtitle + ytitle +
        ref_marker + med_marker + bad_anno + good_anno
    )

    aria = (
        f"Nearest-neighbor cosine margin distribution for the reservoir: a "
        f"histogram of per-item margin = cos(top-1) minus cos(top-2) over "
        f"{n:,} items, x-axis from 0 (near-tie resolution floor) to "
        f"{_fmt(xmax)}. Low-margin near-tie bins are bad-tinted (poor "
        f"resolution), high-margin bins good-tinted (well resolved). An accent "
        f"rule marks the dataset median margin at {_fmt(med)}; a dashed faint "
        f"rule marks the isotropic-reference median at {_fmt(ref_margin)}."
    )

    legend = (
        '<span class="leg">'
        '<span style="color:var(--bad)">▮ near-tie (small margin, crowded)</span> &nbsp; '
        '<span style="color:var(--good)">▮ decisive (large margin, resolved)</span> &nbsp; '
        '<span style="color:var(--accent)">│ median</span> &nbsp; '
        '<span style="color:var(--ink-faint)">┊ isotropic ref</span>'
        '</span>'
    )

    reveal = ("<b>Reveals:</b> how decisively each item's best neighbor beats "
              "its runner-up. Mass piled at the resolution floor means top "
              "matches are near-ties — retrieval is brittle; a fat right tail "
              "means many items have a distinctly separated nearest neighbor.")

    return {"num": "RES 04", "order": 93,
            "name": "Nearest-neighbor cosine margin", "tech": "top1 vs top2",
            "why": "Margin between the top-1 and top-2 cosine neighbor per "
                   "item. A larger margin means the best match is decisively "
                   "separated from the runner-up (well-resolved retrieval); a "
                   "margin near zero is a near-tie where the index can barely "
                   "tell the two apart.",
            "svg": _svg(w, h, aria, body),
            "legend": legend, "reveal": reveal, "cls": ""}
