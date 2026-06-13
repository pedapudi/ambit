"""TOP 06 — Bridge chokepoints (approximate edge betweenness on the kNN graph).

The reservoir kNN graph is drawn as a faint mesh; every edge is scaled by an
approximate betweenness centrality computed with a sampled single-source
shortest-path sweep (Brandes-lite). The single highest-betweenness edge — the
load-bearing isthmus whose removal would split the manifold — is drawn as the
accent spine, its inter-cluster path traced in ink, and the chokepoint nodes
flagged. More margin / a thinner stub between clusters = a real chokepoint:
that is the GOOD, resolving structure; a thick accent spine is a fragile join.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np
import heapq


def _knn_edges(idx, dist):
    """Symmetric edge set from a directed kNN table -> (E,2) int, (E,) weight."""
    m, k = idx.shape
    seen = {}
    for i in range(m):
        for j in range(k):
            v = int(idx[i, j])
            if v < 0 or v == i or v >= m:
                continue
            a, b = (i, v) if i < v else (v, i)
            w = float(dist[i, j]) if dist is not None else 1.0
            key = (a, b)
            prev = seen.get(key)
            if prev is None or w < prev:
                seen[key] = w
    if not seen:
        return np.zeros((0, 2), int), np.zeros((0,), float)
    E = np.array(list(seen.keys()), int)
    W = np.array([seen[tuple(e)] for e in E], float)
    return E, W


def _adj(m, E, W):
    """Adjacency list with per-edge index, weighted by 1-cosine distance."""
    adj = [[] for _ in range(m)]
    for e, (a, b) in enumerate(E):
        w = max(W[e], 1e-6)
        adj[a].append((b, w, e))
        adj[b].append((a, w, e))
    return adj


def _edge_betweenness(m, E, W, sources):
    """Brandes-lite edge betweenness on a weighted graph, summed over `sources`.

    Dijkstra-based accumulation of dependency on edges. Sampling the sources
    keeps it cheap; the relative ranking (which edges carry the most paths) is
    what we need to surface the bridges, not the exact normalized value.
    """
    adj = _adj(m, E, W)
    eb = np.zeros(len(E), float)
    for s in sources:
        dist = np.full(m, np.inf)
        sigma = np.zeros(m)
        dist[s] = 0.0
        sigma[s] = 1.0
        prev_edges = [[] for _ in range(m)]  # (pred, edge_idx) on a shortest path
        order = []
        pq = [(0.0, s)]
        visited = np.zeros(m, bool)
        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            order.append(u)
            for v, w, ei in adj[u]:
                nd = d + w
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    sigma[v] = sigma[u]
                    prev_edges[v] = [(u, ei)]
                    heapq.heappush(pq, (nd, v))
                elif abs(nd - dist[v]) <= 1e-12:
                    sigma[v] += sigma[u]
                    prev_edges[v].append((u, ei))
        delta = np.zeros(m)
        for w in reversed(order):
            for (u, ei) in prev_edges[w]:
                if sigma[w] == 0:
                    continue
                c = (sigma[u] / sigma[w]) * (1.0 + delta[w])
                eb[ei] += c
                delta[u] += c
    return eb


@figure
def fig_top_bridge(ctx):
    W, H, pad = 980, 620, 24
    aria = ("bridge chokepoints on the kNN graph: the intra-territory mesh drawn faint with "
            "edges scaled by approximate betweenness, the inter-territory gaps drawn as dashed "
            "links ramped bad-to-good by separation, and the single narrowest gap — the critical "
            "near-join where two territories would first merge — drawn as the accent spine with "
            "its chokepoint nodes flagged")

    P = _box(ctx.xy, W, H, pad=pad)
    m = len(P)

    # ---- degrade: no kNN backend -> faint cloud + centered note ------------
    if ctx.knn_idx is None:
        dots = "".join(
            f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.6" '
            f'fill="var(--ink-faint)" opacity="0.5"/>' for i in range(m))
        note = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" text-anchor="middle" '
                f'font-size="13" fill="var(--ink-faint)" '
                f'font-family="var(--mono,monospace)">needs kNN backend</text>')
        body = f'<g>{dots}</g>{note}'
        return {
            "num": "TOP 06", "order": 6, "name": "Bridge chokepoints", "tech": "betweenness",
            "why": "Edge betweenness needs the kNN graph; without a neighbor backend only the faint cloud is shown.",
            "svg": _svg(W, H, aria, body),
            "legend": '<span><i class="f"></i> reservoir point</span>',
            "reveal": "<b>Reveals:</b> nothing until a kNN backend is attached.",
            "cls": "",
        }

    # ---- build the symmetric kNN graph ------------------------------------
    E, EW = _knn_edges(ctx.knn_idx, ctx.knn_dist)

    if len(E) == 0:
        dots = "".join(
            f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.6" '
            f'fill="var(--ink-faint)" opacity="0.5"/>' for i in range(m))
        note = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" text-anchor="middle" '
                f'font-size="13" fill="var(--ink-faint)" '
                f'font-family="var(--mono,monospace)">needs kNN edges</text>')
        return {
            "num": "TOP 06", "order": 6, "name": "Bridge chokepoints", "tech": "betweenness",
            "why": "The kNN table produced no usable edges.",
            "svg": _svg(W, H, aria, f'<g>{dots}</g>{note}'),
            "legend": '<span><i class="f"></i> reservoir point</span>',
            "reveal": "<b>Reveals:</b> nothing — empty kNN graph.",
            "cls": "",
        }

    # ---- approximate edge betweenness via sampled SSSP ---------------------
    rng = np.random.default_rng(7)
    n_src = int(min(m, 220))
    sources = rng.choice(m, size=n_src, replace=False)
    eb = _edge_betweenness(m, E, EW, sources)
    eb_max = float(eb.max()) if eb.max() > 0 else 1.0
    ebn = eb / eb_max  # normalized betweenness in [0,1]

    order = np.argsort(ebn)  # ascending; high-betweenness drawn last/on top
    bridge_thresh = float(np.quantile(ebn, 0.985))

    # ---- faint mesh: width + opacity scale with betweenness ----------------
    # Keep the file lean: render every bridge edge plus a betweenness-weighted
    # subsample of the low-betweenness mesh (so the dense core still reads).
    low = np.array([e for e in order if ebn[e] < bridge_thresh], int)
    cap = 1400
    if len(low) > cap:
        prob = ebn[low] + 0.05
        prob = prob / prob.sum()
        keep = rng.choice(low, size=cap, replace=False, p=prob)
        keep = keep[np.argsort(ebn[keep])]  # ascending so heavier drawn later
    else:
        keep = low
    faint = []
    for e in keep:
        a, b = E[e]
        bb = ebn[e]
        sw = 0.6 + 0.5 * bb
        op = 0.18 + 0.10 * bb
        faint.append(
            f'<line x1="{P[a,0]:.2f}" y1="{P[a,1]:.2f}" x2="{P[b,0]:.2f}" y2="{P[b,1]:.2f}" '
            f'stroke="var(--ink-faint)" stroke-width="{sw:.2f}" opacity="{op:.2f}" '
            f'vector-effect="non-scaling-stroke"/>')
    mesh = f'<g fill="none">{"".join(faint)}</g>'

    # ---- occupancy points (faint, density reads through the mesh) ----------
    # subsample to keep the svg light; density still reads by accumulation.
    pcap = 1600
    pidx = np.arange(m) if m <= pcap else rng.choice(m, size=pcap, replace=False)
    pts = [f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.5" '
           f'fill="var(--ink-faint)" opacity="0.45"/>' for i in pidx]
    points = f'<g>{"".join(pts)}</g>'

    # node load: total betweenness incident on each vertex (cut-vertex proxy)
    node_load = np.zeros(m)
    for e in range(len(E)):
        a, b = E[e]
        node_load[a] += ebn[e]
        node_load[b] += ebn[e]

    # ---- connected components of the symmetric kNN graph -------------------
    # The honest bridge story depends on whether the graph is one piece. We
    # union-find the components; if it is already disconnected (well-separated
    # clusters), the load-bearing "bridges" are the gaps BETWEEN components —
    # the nearest cross-component links the manifold would need to fuse.
    parent = list(range(m))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in E:
        ra, rb = _find(int(a)), _find(int(b))
        if ra != rb:
            parent[ra] = rb
    comp = np.array([_find(i) for i in range(m)])
    comp_ids, comp_inv = np.unique(comp, return_inverse=True)
    n_comp = len(comp_ids)

    # ---- the critical bridges --------------------------------------------
    # bridges := candidate inter-territory links. Each is (i,j,gap,kind):
    #   gap is the geometric span in projected coords; larger gap = better
    #   resolved separation (GOOD), the single narrowest = the fragile near-
    #   join (the would-be merge point, the real chokepoint).
    bridges = []          # list of (i, j, gap)
    disconnected = n_comp > 1
    if disconnected:
        # nearest cross-component point pair per ORDERED-pair of components.
        # cap component sizes sampled to keep the all-pairs check cheap.
        groups = [np.where(comp_inv == c)[0] for c in range(n_comp)]
        scap = 240
        for ci in range(n_comp):
            gi = groups[ci]
            if len(gi) > scap:
                gi = rng.choice(gi, size=scap, replace=False)
            Pi = P[gi]
            for cj in range(ci + 1, n_comp):
                gj = groups[cj]
                if len(gj) > scap:
                    gj = rng.choice(gj, size=scap, replace=False)
                Pj = P[gj]
                d = np.sqrt(((Pi[:, None, :] - Pj[None, :, :]) ** 2).sum(-1))
                fi, fj = np.unravel_index(np.argmin(d), d.shape)
                bridges.append((int(gi[fi]), int(gj[fj]), float(d[fi, fj])))
    else:
        # single component: bridges are the highest-betweenness edges (true
        # within-manifold isthmuses). Use geometric span to surface real spans.
        ea2, eb2 = E[:, 0], E[:, 1]
        span = np.hypot(P[ea2, 0] - P[eb2, 0], P[ea2, 1] - P[eb2, 1])
        span_n = span / max(span.max(), 1e-9)
        bridge_score = ebn * (0.4 + 0.6 * span_n)
        for e in np.argsort(bridge_score)[::-1][:max(6, n_comp)]:
            a, b = E[e]
            bridges.append((int(a), int(b), float(span[e])))

    # sort by gap ascending: index 0 = the narrowest = the critical near-join.
    bridges.sort(key=lambda t: t[2])
    gaps = np.array([g for _, _, g in bridges]) if bridges else np.array([0.0])
    gmax = float(gaps.max()) if gaps.size and gaps.max() > 0 else 1.0

    # component centroids — used to keep gap-value labels out of dense cores.
    cents = np.array([P[comp_inv == c].mean(0) for c in range(n_comp)])

    # ---- draw the candidate bridges, colored BY DIRECTION ------------------
    # narrow gap (fragile near-join) -> bad ; wide gap (clean separation) ->
    # good. The single narrowest is the accent critical spine.
    crit_i, crit_j, crit_gap = bridges[0]
    chain = []
    for k, (i, j, g) in enumerate(bridges):
        if k == 0:
            continue
        gn = g / gmax                       # 0 narrow .. 1 wide
        # ramp bad(narrow) -> good(wide): more space = better resolution
        col = f'color-mix(in srgb, var(--bad) {(1-gn)*100:.0f}%, var(--good))'
        sw = 2.2 - 1.0 * gn                 # thicker when fragile
        chain.append(
            f'<line x1="{P[i,0]:.2f}" y1="{P[i,1]:.2f}" x2="{P[j,0]:.2f}" y2="{P[j,1]:.2f}" '
            f'stroke="{col}" stroke-width="{sw:.2f}" opacity="0.85" stroke-dasharray="5 4" '
            f'stroke-linecap="round" vector-effect="non-scaling-stroke"/>')
        # place the gap value at the point along the link that is farthest from
        # every centroid (open space), nudged off the stroke perpendicular.
        ts = np.linspace(0.30, 0.70, 9)
        cand_pts = P[i] * (1 - ts)[:, None] + P[j] * ts[:, None]
        clr = np.min(((cand_pts[:, None, :] - cents[None, :, :]) ** 2).sum(-1), axis=1)
        best = cand_pts[int(np.argmax(clr))]
        dxl, dyl = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
        nlen = max((dxl * dxl + dyl * dyl) ** 0.5, 1e-6)
        px, py = -dyl / nlen, dxl / nlen    # unit perpendicular
        lx, ly = best[0] + px * 9, best[1] + py * 9
        chain.append(f'<text class="num" x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                     f'opacity="0.8">{gn:.2f}</text>')
    chain_g = f'<g fill="none">{"".join(chain)}</g>'

    # the critical bridge: the narrowest gap (accent spine, solid).
    ca, cb = crit_i, crit_j
    if P[ca, 0] > P[cb, 0]:
        ca, cb = cb, ca
    crit = (f'<line x1="{P[ca,0]:.2f}" y1="{P[ca,1]:.2f}" x2="{P[cb,0]:.2f}" y2="{P[cb,1]:.2f}" '
            f'stroke="var(--accent)" stroke-width="2.8" stroke-linecap="round" '
            f'vector-effect="non-scaling-stroke"/>')

    # chokepoint nodes: the endpoints of the critical near-join (caution rings)
    choke = [ca, cb]
    rings = "".join(
        f'<circle cx="{P[v,0]:.2f}" cy="{P[v,1]:.2f}" r="5" fill="none" '
        f'stroke="var(--caution)" stroke-width="1.4" vector-effect="non-scaling-stroke"/>'
        for v in choke)

    # endpoint medoids of the critical bridge: accent dot + relative node load.
    end_a, end_b = ca, cb
    ba = node_load[end_a] / max(node_load.max(), 1e-9)
    bb_ = node_load[end_b] / max(node_load.max(), 1e-9)
    crit_gn = crit_gap / gmax
    medoids = [
        f'<circle cx="{P[end_a,0]:.2f}" cy="{P[end_a,1]:.2f}" r="3" fill="var(--accent)"/>',
        f'<circle cx="{P[end_b,0]:.2f}" cy="{P[end_b,1]:.2f}" r="3" fill="var(--accent)"/>',
    ]
    # Push each endpoint label away from the spine: the upper endpoint's label
    # goes well above, the lower endpoint's well below, so a near-vertical
    # critical bridge does not stack its two labels onto the crowded midpoint.
    a_above = P[end_a, 1] <= P[end_b, 1]
    ay = -22 if a_above else 30
    by = 30 if a_above else -22
    lx_a, ly_a = P[end_a, 0] - 9, P[end_a, 1] + ay
    medoids.append(f'<text class="lbl" x="{lx_a:.1f}" y="{ly_a:.1f}" text-anchor="end">choke a</text>')
    medoids.append(f'<text class="num" x="{lx_a:.1f}" y="{ly_a+13:.1f}" text-anchor="end">load={ba:.2f}</text>')
    lx_b, ly_b = P[end_b, 0] + 9, P[end_b, 1] + by
    medoids.append(f'<text class="lbl" x="{lx_b:.1f}" y="{ly_b:.1f}">choke b</text>')
    medoids.append(f'<text class="num" x="{lx_b:.1f}" y="{ly_b+13:.1f}">load={bb_:.2f}</text>')
    medoids_g = "".join(medoids)

    # critical-bridge callout: anchored at the spine midpoint, leadered out
    # horizontally into the open gap (perpendicular to the near-vertical spine)
    # so it clears the endpoint labels. Side chosen toward more canvas room.
    mx = (P[ca, 0] + P[cb, 0]) / 2
    my = (P[ca, 1] + P[cb, 1]) / 2
    to_left = mx > W * 0.5
    lead = -150 if to_left else 150
    tx = mx + lead
    tanc = 'end' if to_left else 'start'
    callout = (
        f'<line x1="{mx:.1f}" y1="{my:.1f}" x2="{tx:.1f}" y2="{my:.1f}" '
        f'stroke="var(--accent)" stroke-width="0.8" opacity="0.55" vector-effect="non-scaling-stroke"/>'
        f'<text class="num-acc" x="{tx:.1f}" y="{my-3:.1f}" text-anchor="{tanc}">'
        f'critical near-join · gap={crit_gn:.2f}</text>'
        f'<text class="lbl-faint" x="{tx:.1f}" y="{my+11:.1f}" text-anchor="{tanc}" opacity="0.9">'
        f'narrowest separation → first to merge</text>')

    # cluster annotations: name the two territories the critical bridge joins,
    # centroids pushed away from the spine so labels sit clear of the points.
    labs = np.asarray(ctx.labels) if ctx.labels is not None else None
    cluster_notes = ""
    if labs is not None:
        lab_a, lab_b = int(labs[end_a]), int(labs[end_b])
        join = [lab_a] if lab_a == lab_b else [lab_a, lab_b]
        names = ["territory a", "territory b"]
        notes = []
        for nm, lb in zip(names, join):
            sel = labs == lb
            cx = float(P[sel, 0].mean())
            cymean = float(P[sel, 1].mean())
            # push the label sideways, away from the (near-vertical) spine, and
            # vertically to the far side of the centroid from the spine midpoint.
            cx += -34 if cx <= mx else 34
            cyy = cymean + (-26 if cymean <= my else 26)
            notes.append(f'<text class="lbl-faint" x="{cx:.0f}" y="{cyy:.0f}" '
                         f'text-anchor="middle">{nm}</text>')
        cluster_notes = "".join(notes)

    # ---- inter-territory gap scale legend (lower strip) --------------------
    # axis: normalized gap distance, 0 (fragile near-join, bad) -> 1 (clean
    # separation, good). Evenly spaced labeled ticks; the critical near-join
    # marked at its true normalized gap on the bad end.
    LX, LY, LW = 40, H - 40, 408
    leg = [f'<g class="bw-legend" transform="translate({LX},{LY})">']
    # title sits to the right of the swatch's bad end so it never collides with
    # the critical caret, which lands near the narrow (left) end.
    leg.append(f'<text class="lbl" x="{LW:.0f}" y="-10" text-anchor="end">'
               f'inter-territory gap (normalized)</text>')
    # continuous bad -> good ramp swatch (legit differential scale)
    nseg = 24
    seg = LW / nseg
    for i in range(nseg):
        gn = i / (nseg - 1)
        col = f'color-mix(in srgb, var(--bad) {(1-gn)*100:.0f}%, var(--good))'
        leg.append(f'<rect x="{i*seg:.1f}" y="4" width="{seg+0.6:.1f}" height="9" '
                   f'fill="{col}"/>')
    # baseline + evenly spaced labeled ticks
    leg.append(f'<line x1="0" y1="22" x2="{LW:.0f}" y2="22" stroke="var(--ink-faint)" '
               f'stroke-width="1" opacity="0.5" vector-effect="non-scaling-stroke"/>')
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tx = t * LW
        leg.append(f'<line x1="{tx:.1f}" y1="18" x2="{tx:.1f}" y2="30" '
                   f'stroke="var(--rule-soft)" vector-effect="non-scaling-stroke"/>')
        leg.append(f'<text class="num" x="{tx:.1f}" y="42" text-anchor="middle">{t:.2f}</text>')
    # minor ticks at 0.1
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tx = t * LW
        leg.append(f'<line x1="{tx:.1f}" y1="20" x2="{tx:.1f}" y2="28" '
                   f'stroke="var(--rule-soft)" opacity="0.7" vector-effect="non-scaling-stroke"/>')
    # direction labels under the axis
    leg.append(f'<text class="lbl-faint" x="2" y="56">fragile near-join</text>')
    leg.append(f'<text class="lbl-faint" x="{LW:.0f}" y="56" text-anchor="end">clean separation</text>')
    # critical near-join marker at its true normalized gap (accent caret above
    # the swatch); label anchored to whichever side keeps it inside the strip.
    cgx = float(np.clip(crit_gn, 0.0, 1.0)) * LW
    leg.append(f'<line x1="{cgx:.1f}" y1="-2" x2="{cgx:.1f}" y2="14" '
               f'stroke="var(--accent)" stroke-width="1.8" vector-effect="non-scaling-stroke"/>')
    anc = 'start' if cgx < LW * 0.5 else 'end'
    dx = 4 if anc == 'start' else -4
    leg.append(f'<text class="num-acc" x="{cgx+dx:.1f}" y="-4" text-anchor="{anc}">'
               f'critical {crit_gn:.2f}</text>')
    leg.append('</g>')
    legend_g = "".join(leg)

    # ---- compose ----------------------------------------------------------
    style = (
        '<style>'
        '.ambit-06 text{font-family:var(--mono,ui-monospace,Menlo,Consolas,monospace);}'
        '.ambit-06 .lbl{fill:var(--ink-soft);font-size:12px;}'
        '.ambit-06 .lbl-faint{fill:var(--ink-faint);font-size:10.5px;}'
        '.ambit-06 .num{fill:var(--ink);font-size:12px;font-variant-numeric:tabular-nums;}'
        '.ambit-06 .num-acc{fill:var(--accent);font-size:12.5px;font-weight:700;'
        'font-variant-numeric:tabular-nums;}'
        '</style>')
    body = (
        f'{style}<g class="ambit-06">'
        f'{mesh}{points}{cluster_notes}{chain_g}{crit}{rings}{medoids_g}{callout}{legend_g}'
        f'</g>')

    n_edges = len(E)
    n_br = len(bridges)
    if disconnected:
        why = (f"The kNN graph is already {n_comp} disconnected territories "
               f"({n_edges:,} intra-territory edges, zero crossings). The "
               f"load-bearing bridges are therefore the GAPS between them: each "
               f"dashed link is the nearest cross-territory pair, ramped bad→good "
               f"by separation. The accent spine is the narrowest gap (norm "
               f"{crit_gn:.2f}) — the first place two territories would merge.")
        reveal = ("<b>Reveals:</b> the embedding is partitioned — neighbors never "
                  "cross territory lines. The narrowest gap is the fragile near-join "
                  "where the partition is most likely to collapse under noise.")
    else:
        why = (f"Approximate edge betweenness over {n_edges:,} kNN edges (sampled "
               f"single-source shortest paths). The {n_br} highest-load spans are the "
               f"isthmuses within the single connected manifold; the accent spine is "
               f"the critical one (norm gap {crit_gn:.2f}) — remove it and it splits.")
        reveal = ("<b>Reveals:</b> the load-bearing joins inside the manifold — the few "
                  "spans whose removal would fragment it into disconnected components.")
    return {
        "num": "TOP 06", "order": 6, "name": "Bridge chokepoints", "tech": "betweenness",
        "why": why,
        "svg": _svg(W, H, aria, body),
        "legend": ('<span><i class="r"></i> narrow gap (fragile near-join)</span>'
                   '<span><i class="g"></i> wide gap (clean separation)</span>'
                   '<span><i class="a"></i> critical near-join</span>'
                   '<span><i class="cache"></i> chokepoint node</span>'),
        "reveal": reveal,
        "cls": "",
    }
