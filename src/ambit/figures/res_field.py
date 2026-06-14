"""RES 06 — Local concentration field. The unsupervised, multiscale local-crowding
measure rendered as its native artifact: the DISTRIBUTION of each item's local
neighborhood concentration (mean cosine to its k nearest, at the most-separated
scale), drawn against a synthetic isotropic reference. The shape is the read —

  - one mode near the isotropic reference   -> roomy / uniform
  - the whole mode shifted far above it      -> globally crowded (a union of cones)
  - a separated high mode across a valley     -> a crowded pocket (one per bump)

All numbers come from `local_anisotropy.for_ctx` (the generation step); this figure
only draws them. See docs/concepts/cluster-sensitive-anisotropy.html.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg
from .. import local_anisotropy


def _fmt(v):
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s or "0"


@figure
def fig_res_field(ctx):
    w, h = 760, 470
    L, R, T, B = 92, 732, 92, 392

    la = local_anisotropy.for_ctx(ctx)
    field = np.asarray(la.field, float)
    iso = np.asarray(la.iso_ref[la.scale_star], float)
    n = field.size
    bulk, iso_bulk = float(la.bulk), float(la.iso_bulk)
    gc = float(la.global_crowding)
    valley = la.valley

    # ---- shared axis 0 .. nice ceiling covering both clouds
    hi = max(float(field.max()), float(iso.max())) * 1.04
    xmax = float(np.ceil(hi * 10.0) / 10.0) or 1.0
    nb = 72
    edges = np.linspace(0.0, xmax, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    d_counts, _ = np.histogram(np.clip(field, 0, xmax - 1e-9), bins=edges)
    i_counts, _ = np.histogram(np.clip(iso, 0, xmax - 1e-9), bins=edges)
    cmax = max(int(d_counts.max()), 1)
    iref_peak = max(int(i_counts.max()), 1)            # isotropic ref is peak-scaled (own n)

    def X(v):
        return L + (v / xmax) * (R - L)

    def Y(c):
        return B - (c / cmax) * (B - T)

    # ---- decide the read (long, for aria/reveal) and a short tag (for the subtitle)
    crowded = gc > 8.0
    if la.multimodal and valley is not None:
        np_ = len(la.pockets)
        tag = f"{np_} crowded pocket{'s' if np_ != 1 else ''}"
        read = f"a crowded mode separates from the bulk — {tag}"
    elif crowded:
        tag = "globally crowded (union of cones)"
        read = "the whole field sits far above the isotropic reference — globally crowded (a union of cones)"
    else:
        tag = "roomy / uniform"
        read = "one mode near the isotropic reference — roomy, well-resolved"

    def tone(center):
        if la.multimodal and valley is not None:
            return "--bad" if center > valley else "--good"
        return "--caution" if crowded else "--good"

    pbw = (R - L) / nb
    bars = []
    for i in range(nb):
        c = int(d_counts[i])
        if c <= 0:
            continue
        x0, y = X(edges[i]), Y(c)
        bars.append(f'<rect x="{x0 + 0.4:.2f}" y="{y:.1f}" width="{max(pbw - 0.8, 0.5):.2f}" '
                    f'height="{B - y:.1f}" fill="color-mix(in srgb, var({tone(centers[i])}) 58%, var(--panel))"/>')
    bars_svg = "<g>" + "".join(bars) + "</g>"

    # ---- isotropic reference, drawn as a faint ghost outline peak-scaled to the plot
    pts = []
    for i in range(nb):
        yy = B - (i_counts[i] / iref_peak) * (B - T) * 0.92
        pts.append(f"{X(centers[i]):.1f},{yy:.1f}")
    iso_poly = (f'<polyline points="{X(0):.1f},{B:.1f} ' + " ".join(pts) +
                f' {X(centers[-1]):.1f},{B:.1f}" fill="color-mix(in srgb, var(--ink-faint) 14%, transparent)" '
                f'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 2" stroke-opacity="0.8"/>')

    # ---- gridlines / x ticks every 0.1
    ticks = [round(0.1 * i, 1) for i in range(int(xmax / 0.1) + 1)]
    grid = "".join(f'<line x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{B}"/>' for t in ticks if 0 < t < xmax)
    grid_svg = f'<g stroke="var(--rule-soft)" stroke-width="0.6">{grid}</g>'
    major = "".join(f'<line x1="{X(t):.1f}" y1="{B}" x2="{X(t):.1f}" y2="{B + 6}"/>' for t in ticks)
    tlabels = "".join(f'<text x="{X(t):.1f}" y="{B + 18}">{_fmt(t)}</text>' for t in ticks)
    xaxis_svg = (f'<g stroke="var(--ink-faint)" stroke-width="0.9">{major}</g>'
                 f'<g font-size="9" fill="var(--ink-soft)" text-anchor="middle" '
                 f'style="font-variant-numeric:tabular-nums">{tlabels}</g>')

    yticks = sorted({0, int(round(cmax * 0.5)), cmax})
    ylabels = "".join(f'<text x="{L - 8}" y="{Y(c) + 3:.1f}" font-size="9" fill="var(--ink-faint)" '
                      f'text-anchor="end">{c:,}</text>' for c in yticks)
    baseline = f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>'

    # ---- markers: isotropic-reference median (dashed faint) + dataset bulk (accent)
    iso_x, bulk_x = X(min(iso_bulk, xmax)), X(min(bulk, xmax))
    iso_mark = (f'<line x1="{iso_x:.1f}" y1="{T}" x2="{iso_x:.1f}" y2="{B}" stroke="var(--ink-faint)" '
                f'stroke-width="1.4" stroke-dasharray="3 3"/>'
                f'<text x="{iso_x + 5:.1f}" y="{T + 13}" font-size="9.5" fill="var(--ink-faint)" '
                f'text-anchor="start">isotropic ref = {_fmt(iso_bulk)}</text>')
    bulk_anchor = "start" if bulk_x < R - 160 else "end"
    bulk_dx = 6 if bulk_anchor == "start" else -6
    bulk_mark = (f'<line x1="{bulk_x:.1f}" y1="{T}" x2="{bulk_x:.1f}" y2="{B}" stroke="var(--accent)" '
                 f'stroke-width="2.2"/><circle cx="{bulk_x:.1f}" cy="{T}" r="3" fill="var(--accent)"/>'
                 f'<text x="{bulk_x + bulk_dx:.1f}" y="{T - 6}" font-size="10.5" font-weight="700" '
                 f'fill="var(--accent)" text-anchor="{bulk_anchor}">dataset bulk = {_fmt(bulk)}</text>')

    # gap bracket between the two medians = the global-crowding story
    gap_y = T + 30
    gap = ""
    if bulk_x - iso_x > 40:
        gap = (f'<line x1="{iso_x:.1f}" y1="{gap_y}" x2="{bulk_x:.1f}" y2="{gap_y}" stroke="var(--ink-soft)" '
               f'stroke-width="1" marker-start="url(#rfA)" marker-end="url(#rfA)"/>'
               f'<text x="{(iso_x + bulk_x) / 2:.1f}" y="{gap_y - 5}" font-size="9.5" fill="var(--ink-soft)" '
               f'text-anchor="middle">global crowding = {gc:.0f}σ</text>'
               f'<defs><marker id="rfA" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto">'
               f'<path d="M6,1 L1,3.5 L6,6" fill="none" stroke="var(--ink-soft)" stroke-width="1"/></marker></defs>')

    # pocket flags above their bumps
    pmarks = ""
    for p in la.pockets[:6]:
        px = X(min(float(p.concentration), xmax))
        pmarks += (f'<line x1="{px:.1f}" y1="{T + 44}" x2="{px:.1f}" y2="{B}" stroke="var(--bad)" '
                   f'stroke-width="1" stroke-dasharray="2 2" opacity="0.7"/>'
                   f'<text x="{px:.1f}" y="{T + 41}" font-size="8.5" fill="var(--bad)" '
                   f'text-anchor="middle">pocket · n={p.size:,}</text>')

    title = (f'<text x="{L}" y="26" font-size="13" font-weight="700" fill="var(--ink)">'
             f'local concentration field — mean cosine to the {la.scale_star} nearest, per item</text>'
             f'<text x="{R}" y="26" font-size="10.5" fill="var(--ink-faint)" text-anchor="end">{n:,} items</text>'
             f'<text x="{L}" y="42" font-size="10.5" fill="var(--ink-faint)">'
             f'scales {", ".join(map(str, la.scales))} · headline k = {la.scale_star} (most separated) · '
             f'{tag}</text>')

    ytitle = (f'<text x="22" y="{(T + B) / 2:.1f}" font-size="8.5" fill="var(--ink-faint)" text-anchor="middle" '
              f'transform="rotate(-90 22 {(T + B) / 2:.1f})">items per bin</text>')
    xtitle = (f'<text x="{R}" y="{B + 32}" font-size="9" fill="var(--ink-faint)" text-anchor="end">'
              f'local concentration  =  mean cos to k-NN  (0 isotropic … 1 collapsed)</text>')
    rmark = (f'<text x="{L}" y="{B + 32}" font-size="8.5" fill="var(--ink-faint)" text-anchor="start">'
             f'isotropic reference scaled to its own peak</text>')

    body = (title + grid_svg + iso_poly + bars_svg + baseline + xaxis_svg + ylabels +
            ytitle + xtitle + rmark + gap + iso_mark + bulk_mark + pmarks)

    aria = (f"Distribution of the per-item local concentration field over {n:,} reservoir items at scale "
            f"k={la.scale_star}: a histogram from 0 (isotropic) to {_fmt(xmax)} (collapsed). The dataset bulk "
            f"sits at {_fmt(bulk)}, an accent rule; the synthetic isotropic reference, a faint dashed ghost, "
            f"peaks near {_fmt(iso_bulk)}. The {gc:.0f}-sigma gap between them is the global crowding. {read}.")

    legend = ('<span class="leg">'
              + ('<span style="color:var(--bad)">▮ crowded pocket</span> &nbsp; '
                 '<span style="color:var(--good)">▮ bulk</span> &nbsp; ' if la.multimodal else
                 (f'<span style="color:var(--caution)">▮ field (globally crowded)</span> &nbsp; '
                  if crowded else '<span style="color:var(--good)">▮ field (roomy)</span> &nbsp; '))
              + '<span style="color:var(--accent)">│ dataset bulk</span> &nbsp; '
                '<span style="color:var(--ink-faint)">┊ isotropic reference</span></span>')

    reveal = ("<b>Reveals:</b> whether crowding is <em>global</em> (the whole field shifted above the "
              "isotropic reference — a union of locally-collapsed cones) or <em>local</em> (a separated "
              "high mode = a crowded pocket). A single scalar cannot tell these apart; the distribution can.")

    return {"num": "RES 06", "order": 95,
            "name": "Local concentration field", "tech": "k-NN concentration · multiscale",
            "why": "For each item, the mean cosine to its k nearest neighbors — how tightly its local "
                   "neighborhood collapses — measured at several scales and shown as a distribution against a "
                   "synthetic isotropic reference. Crowding is local and non-collapsible over clusters, so the "
                   "shape of this field, not a global average, is the honest read.",
            "svg": _svg(w, h, aria, body),
            "legend": legend, "reveal": reveal, "cls": ""}
