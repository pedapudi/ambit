from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_top_knn(ctx):
    """kNN manifold graph over the reservoir: hairline neighbor edges graded by
    projected length, faint nodes, with the highest-degree hub (and its spokes)
    carrying the single accent emphasis."""
    w, h, pad = 760, 470, 22
    P = _box(ctx.xy, w, h, pad=pad)
    m = len(P)

    idx = ctx.knn_idx
    if idx is None:
        # degrade: faint cloud + centered note, never crash
        dots = "".join(
            f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.6" '
            f'fill="var(--ink-faint)" fill-opacity="0.4"/>'
            for i in range(m)
        )
        note = (f'<text x="{w/2:.0f}" y="{h/2:.0f}" text-anchor="middle" '
                f'font-size="13" fill="var(--ink-faint)">needs kNN backend</text>')
        return {
            "num": "TOP 05", "order": 5, "name": "kNN manifold graph", "tech": "knn graph",
            "why": "The reservoir kNN graph wires each point to its nearest neighbors; "
                   "the wiring needs a kNN backend that was not available for this scan.",
            "svg": _svg(w, h, "kNN manifold graph — needs kNN backend", dots + note),
            "legend": '<span><i class="f"></i> reservoir point</span>'
                      '<span><i class="dash"></i> needs kNN backend</span>',
            "reveal": "<b>Reveals:</b> the neighbor wiring of the manifold — unavailable without a kNN backend.",
            "cls": "",
        }

    k = idx.shape[1]
    # ------------------------------------------------------------- build edges
    # one undirected edge per directed neighbor link, de-duplicated by (min,max)
    seen = set()
    edges = []  # (a, b, length_in_px)
    deg = np.zeros(m, dtype=int)
    for a in range(m):
        for b in idx[a]:
            b = int(b)
            if b == a or b < 0 or b >= m:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            d = float(np.hypot(P[a, 0] - P[b, 0], P[a, 1] - P[b, 1]))
            edges.append((a, b, d))
            deg[a] += 1
            deg[b] += 1

    if edges:
        lens = np.array([e[2] for e in edges])
        lmax = float(lens.max()) or 1.0
    else:
        lens = np.array([1.0])
        lmax = 1.0

    hub = int(np.argmax(deg))  # highest-degree node = densest local hub
    hub_set = set()
    for a, b, _ in edges:
        if a == hub or b == hub:
            hub_set.add((a, b))

    # Thin the mesh for a legible, lightweight skeleton: a full 4000-node, k-wired
    # graph is an ink blot. The manifold structure lives in the SHORT links, so keep
    # the shortest-edge backbone (every node still appears as a node below). Hub
    # spokes are always retained so the accent reads in full.
    MAX_EDGES = 4500
    if len(edges) > MAX_EDGES:
        order = np.argsort(lens)  # shortest first
        keep = set(order[:MAX_EDGES].tolist())
        edges = [edges[i] for i in range(len(edges)) if i in keep or
                 ((edges[i][0], edges[i][1]) in hub_set)]

    # ------------------------------------------------------------- draw edges
    # MORE space / longer neighbor edge = sparser, higher resolution -> read by
    # opacity grade (short=opaque mesh knot, long=faint reach). Hairline ink-faint.
    body = ['<g aria-label="kNN edges: hairline var(--ink-faint), opacity graded by '
            'projected length; short=opaque mesh, long=faint reach">']
    for a, b, d in edges:
        if (a, b) in hub_set:
            continue  # drawn later in accent so the hub reads on top
        t = d / lmax  # 0 short .. 1 long
        op = 0.50 - 0.34 * t        # short edges denser/opaque, long edges faint
        sw = 0.85 - 0.30 * t
        body.append(
            f'<path d="M{P[a,0]:.1f} {P[a,1]:.1f}L{P[b,0]:.1f} {P[b,1]:.1f}" '
            f'stroke="var(--ink-faint)" stroke-width="{sw:.2f}" '
            f'stroke-opacity="{op:.3f}" fill="none" vector-effect="non-scaling-stroke"/>'
        )
    body.append('</g>')

    # ------------------------------------------------------------- draw nodes
    body.append('<g aria-label="reservoir nodes, faint ink">')
    for i in range(m):
        if i == hub:
            continue
        body.append(
            f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.4" '
            f'fill="var(--ink-faint)" fill-opacity="0.55"/>'
        )
    body.append('</g>')

    # ---------------------------------------------------- accent: the hub + spokes
    nbrs = sorted(hub_set, key=lambda e: -(e[0] == hub) - (e[1] == hub))
    body.append(f'<g aria-label="highest-degree hub (degree {int(deg[hub])}) and its spokes, accent">')
    for a, b in hub_set:
        body.append(
            f'<path d="M{P[a,0]:.1f} {P[a,1]:.1f}L{P[b,0]:.1f} {P[b,1]:.1f}" '
            f'stroke="var(--accent)" stroke-width="1.4" stroke-opacity="0.92" '
            f'fill="none" vector-effect="non-scaling-stroke"/>'
        )
    # the neighbor endpoints of the hub, slightly emphasized
    for a, b in hub_set:
        o = b if a == hub else a
        body.append(
            f'<circle cx="{P[o,0]:.1f}" cy="{P[o,1]:.1f}" r="2.0" '
            f'fill="var(--accent)" fill-opacity="0.55"/>'
        )
    body.append(
        f'<circle cx="{P[hub,0]:.1f}" cy="{P[hub,1]:.1f}" r="3.6" fill="var(--accent)"/>'
        f'<circle cx="{P[hub,0]:.1f}" cy="{P[hub,1]:.1f}" r="6.0" fill="none" '
        f'stroke="var(--accent)" stroke-width="1" stroke-opacity="0.45"/>'
    )
    body.append('</g>')

    # ------------------------------------------------------------- annotations
    hx = min(max(P[hub, 0], 70), w - 70)
    hy_lab = P[hub, 1] - 12 if P[hub, 1] > 40 else P[hub, 1] + 20
    body.append(
        f'<text x="{hx:.1f}" y="{hy_lab:.1f}" text-anchor="middle" font-size="10.5" '
        f'font-weight="700" fill="var(--accent)">hub · degree {int(deg[hub])}</text>'
    )

    # ----------------------------------------------- edge-length legend (ticks)
    # mono-labeled, evenly spaced ticks on the quantitative (edge-length) axis
    lx0, lx1, ly = 28.0, 168.0, 30.0
    body.append(f'<text x="{lx0:.1f}" y="{ly-8:.1f}" font-size="9.5" fill="var(--ink-soft)" '
                f'font-weight="600">edge length → hairline opacity</text>')
    for f_ in (0.0, 0.5, 1.0):
        xx = lx0 + f_ * (lx1 - lx0)
        op = 0.50 - 0.34 * f_
        sw = 0.85 - 0.30 * f_
        seglen = 30.0
        body.append(
            f'<path d="M{xx:.1f} {ly:.1f}L{xx+seglen:.1f} {ly:.1f}" stroke="var(--ink-faint)" '
            f'stroke-width="{sw:.2f}" stroke-opacity="{op:.3f}" fill="none" '
            f'vector-effect="non-scaling-stroke"/>'
        )
    # axis rule + evenly spaced labeled px ticks
    ay = ly + 14
    body.append(f'<line x1="{lx0:.1f}" y1="{ay:.1f}" x2="{lx1:.1f}" y2="{ay:.1f}" '
                f'stroke="var(--rule-soft)" stroke-width="0.7" vector-effect="non-scaling-stroke"/>')
    for f_ in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx = lx0 + f_ * (lx1 - lx0)
        val = f_ * lmax
        body.append(f'<line x1="{xx:.1f}" y1="{ay:.1f}" x2="{xx:.1f}" y2="{ay+3.5:.1f}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.7" vector-effect="non-scaling-stroke"/>')
        body.append(f'<text x="{xx:.1f}" y="{ay+13:.1f}" text-anchor="middle" font-size="7.5" '
                    f'fill="var(--ink-faint)" style="font-variant-numeric:tabular-nums">{val:.0f}</text>')
    body.append(f'<text x="{lx1+6:.1f}" y="{ay+13:.1f}" font-size="7.5" fill="var(--ink-faint)" '
                f'style="font-variant-numeric:tabular-nums">px</text>')

    # corner caption (mirrors template header band)
    body.append(f'<text x="{w-28:.0f}" y="22" text-anchor="end" font-size="11" '
                f'font-weight="700" fill="var(--accent)">accent = highest-degree hub + spokes</text>')
    body.append(f'<text x="{w-28:.0f}" y="38" text-anchor="end" font-size="10" '
                f'fill="var(--ink-faint)">k={k} · {len(edges):,} edges drawn · {m:,} nodes</text>')

    med_deg = float(np.median(deg))
    aria = (f"kNN manifold graph of the reservoir: {m} nodes wired by {len(edges)} hairline "
            f"k={k} neighbor edges over the PCA projection, opacity graded by projected length; "
            f"the highest-degree hub (degree {int(deg[hub])}) and its spokes carry the single accent")
    return {
        "num": "TOP 05", "order": 5, "name": "kNN manifold graph", "tech": "knn graph",
        "why": f"Each of the {m:,} reservoir points is wired to its {k} nearest neighbors over the "
               f"projection; hairline edges grade by length (short = mesh knot, long = reach) and the "
               f"densest hub (degree {int(deg[hub])}, median {med_deg:.0f}) takes the single accent.",
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": '<span><i class="f"></i> kNN edge (opacity = length)</span>'
                  '<span><i class="f"></i> reservoir node</span>'
                  '<span><i class="a"></i> highest-degree hub + spokes</span>',
        "reveal": "<b>Reveals:</b> the connectivity skeleton of the manifold — where neighbor wiring "
                  "knots into dense hubs versus where long reaching edges bridge sparse, high-resolution regions.",
        "cls": "",
    }
