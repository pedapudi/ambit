from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np

# Fixed camera shared with 3D 01: azimuth 35 deg, elevation 22 deg.
_AZ = np.radians(35.0)
_EL = np.radians(22.0)


def _camera(xyz):
    """Project (N,3) world points to camera space.

    Returns screen coords (N,2) in a right/down convention plus a camera-depth
    array (N,) where larger = nearer the viewer. Uses an azimuth+elevation
    orbit identical to the 3D 01 cloud so the two figures register.
    """
    p = np.asarray(xyz, float)
    ca, sa = np.cos(_AZ), np.sin(_AZ)
    ce, se = np.cos(_EL), np.sin(_EL)
    # azimuth about world-up (z), then tilt by elevation
    right = np.array([ca, -sa, 0.0])
    up = np.array([-sa * se, -ca * se, ce])
    fwd = np.array([sa * ce, ca * ce, se])  # points from scene toward camera
    sx = p @ right
    sy = p @ up
    depth = p @ fwd  # larger -> nearer
    return np.column_stack([sx, sy]), depth


@figure
def fig_d3_mesh(ctx):
    w, h, pad = 820, 500, 40

    # ---- degrade if no neighbor backend -------------------------------------
    if ctx.knn_idx is None or ctx.xyz is None:
        P = _box(ctx.xy, w, h)
        dots = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.2" fill="var(--ink-faint)" '
            f'fill-opacity="0.4"/>' for x, y in P
        )
        note = (f'<text x="{w/2:.0f}" y="{h/2:.0f}" font-size="12" '
                f'fill="var(--ink-faint)" text-anchor="middle" '
                f'font-family="ui-monospace,monospace">needs kNN backend</text>')
        return {
            "num": "3D 04", "order": 23, "name": "kNN mesh in 3-space", "tech": "knn · 3d",
            "why": "Neighbor graph drawn in projected 3-space; the backend is unavailable so only the faint cloud is shown.",
            "svg": _svg(w, h, "kNN mesh in projected 3-space (neighbor backend unavailable)", dots + note),
            "legend": '<span><i class="f"></i> point (no edges)</span>',
            "reveal": "<b>Reveals:</b> nothing yet — supply a kNN backend to draw the mesh.",
            "cls": "",
        }

    # ---- camera projection (fixed orbit, same as 3D 01) ---------------------
    S, depth = _camera(ctx.xyz)
    m = S.shape[0]
    P = _box(S, w, h, pad)  # screen coords in the svg box (y already flipped)

    # depth -> [0,1], 1 = nearest the camera
    dn = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)

    k = min(6, ctx.knn_idx.shape[1])  # k=6 cosine edges, like the study figure
    idx = np.asarray(ctx.knn_idx[:, :k])
    dist = (np.asarray(ctx.knn_dist[:, :k]) if ctx.knn_dist is not None
            else np.zeros_like(idx, float))

    # unique undirected edges -------------------------------------------------
    src = np.repeat(np.arange(m), k)
    dst = idx.reshape(-1)
    ed = dist.reshape(-1)
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    key = a.astype(np.int64) * m + b
    _, uniq = np.unique(key, return_index=True)
    a, b, ed = a[uniq], b[uniq], ed[uniq]
    valid = a != b
    a, b, ed = a[valid], b[valid], ed[valid]

    # ---- the accent "bridge": the load-bearing long edge --------------------
    # A neighbor edge that spans the largest projected distance is the thin
    # filament whose two ends sit in different parts of the cloud -- cutting it
    # would split a knot off the core. Larger span = more room resolved = GOOD.
    span = np.hypot(P[a, 0] - P[b, 0], P[a, 1] - P[b, 1])
    # combine projected span with cosine distance so we reward genuine reach
    reach = span * (1.0 + ed)
    bridge_e = int(np.argmax(reach)) if len(reach) else -1
    # grow a short accent spine around that edge by following each endpoint's
    # nearest neighbor a couple of hops -> a small depth-spanning filament
    spine = set()
    if bridge_e >= 0:
        ai, bi = int(a[bridge_e]), int(b[bridge_e])
        spine_nodes = [ai, bi]
        for seed in (ai, bi):
            cur = seed
            for _ in range(3):
                nxt = int(idx[cur, 0])
                if nxt == cur:
                    break
                lo, hi = min(cur, nxt), max(cur, nxt)
                spine.add(lo * m + hi)
                spine_nodes.append(nxt)
                cur = nxt
        spine.add(min(ai, bi) * m + max(ai, bi))
    else:
        spine_nodes = []

    # ---- draw edges, far first so near strokes overlay ----------------------
    edepth = 0.5 * (dn[a] + dn[b])
    order = np.argsort(edepth)  # ascending depth: far -> near
    # cap the rendered edge set so the svg stays light; keep every spine edge
    # plus a depth-uniform sample of the rest (preserves near/far balance).
    MAX_EDGES = 2600
    if len(order) > MAX_EDGES:
        spine_mask = np.array(
            [(min(int(a[j]), int(b[j])) * m + max(int(a[j]), int(b[j]))) in spine
             for j in order], dtype=bool)
        keep = np.zeros(len(order), dtype=bool)
        keep[spine_mask] = True
        budget = MAX_EDGES - int(spine_mask.sum())
        rest = np.where(~spine_mask)[0]
        if budget > 0 and len(rest) > budget:
            pick = np.linspace(0, len(rest) - 1, budget).astype(int)
            keep[rest[pick]] = True
        elif budget > 0:
            keep[rest] = True
        order = order[keep]
    edge_svg = []
    accent_svg = []
    for j in order:
        i0, i1 = int(a[j]), int(b[j])
        x1, y1 = P[i0]
        x2, y2 = P[i1]
        d = 0.5 * (dn[i0] + dn[i1])  # 1 = near
        ekey = min(i0, i1) * m + max(i0, i1)
        if ekey in spine:
            # accent spine: near = bolder/brighter, far = thinner
            sw = 1.4 + 0.9 * d
            op = 0.78 + 0.16 * d
            accent_svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="var(--accent)" stroke-width="{sw:.2f}" '
                f'stroke-opacity="{op:.2f}" vector-effect="non-scaling-stroke"/>')
        else:
            # depth-faded hairline cosine edge
            op = 0.10 + 0.34 * d
            sw = 0.55 + 0.35 * d
            edge_svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="var(--ink-faint)" stroke-width="{sw:.2f}" '
                f'stroke-opacity="{op:.2f}" vector-effect="non-scaling-stroke"/>')

    # ---- nodes: small depth-graded dots under nothing (drawn over edges) ----
    node_order = np.argsort(dn)  # far first
    MAX_NODES = 1800
    if len(node_order) > MAX_NODES:
        pick = np.linspace(0, len(node_order) - 1, MAX_NODES).astype(int)
        node_order = node_order[pick]
    node_svg = []
    for i in node_order:
        d = dn[i]
        r = 0.8 + 1.1 * d
        op = 0.18 + 0.40 * d
        node_svg.append(
            f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="{r:.2f}" '
            f'fill="var(--ink-faint)" fill-opacity="{op:.2f}"/>')

    # accent endpoints emphasised
    end_svg = []
    if bridge_e >= 0:
        for i in (int(a[bridge_e]), int(b[bridge_e])):
            end_svg.append(
                f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="3.0" '
                f'fill="var(--accent)" fill-opacity="0.95"/>')

    # ---- corner gnomon: x/y/z axes with near/far depth ticks ----------------
    gx, gy = pad + 18, h - pad - 18  # gnomon origin (lower-left)
    L = 70.0
    # project unit world axes through the same camera, anchored at origin
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    ax_s, _ = _camera(axes)
    org_s, _ = _camera(np.zeros((1, 3)))
    gn = []
    names = ("x", "y", "z")
    for c in range(3):
        vx = ax_s[c, 0] - org_s[0, 0]
        vy = ax_s[c, 1] - org_s[0, 1]
        n = np.hypot(vx, vy) or 1.0
        ux, uy = vx / n, -vy / n  # flip y to svg-down
        ex, ey = gx + ux * L, gy + uy * L
        gn.append(
            f'<line x1="{gx:.1f}" y1="{gy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="var(--ink-faint)" stroke-width="0.7" vector-effect="non-scaling-stroke"/>')
        # evenly spaced ticks along the axis (near .. far)
        for t in (0.33, 0.66, 1.0):
            tx, ty = gx + ux * L * t, gy + uy * L * t
            # perpendicular tick mark
            px, py = -uy * 3.0, ux * 3.0
            gn.append(
                f'<line x1="{tx-px:.1f}" y1="{ty-py:.1f}" x2="{tx+px:.1f}" y2="{ty+py:.1f}" '
                f'stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke"/>')
        gn.append(
            f'<text x="{ex+ux*8:.1f}" y="{ey+uy*8+3:.1f}" font-size="9" '
            f'font-family="ui-monospace,monospace" fill="var(--ink-faint)" '
            f'text-anchor="middle">{names[c]}</text>')
    gn.append(
        f'<text x="{gx-4:.1f}" y="{gy+12:.1f}" font-size="7.5" '
        f'font-family="ui-monospace,monospace" fill="var(--ink-faint)" '
        f'text-anchor="end">near</text>')

    # ---- callouts -----------------------------------------------------------
    callouts = []
    if spine_nodes:
        bi = int(a[bridge_e]) if bridge_e >= 0 else spine_nodes[0]
        bx, by = P[bi]
        ty = by - 22 if by > 60 else by + 22
        callouts.append(
            f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{bx:.1f}" y2="{ty:.1f}" '
            f'stroke="var(--accent)" stroke-width="0.7" stroke-opacity="0.6" '
            f'vector-effect="non-scaling-stroke"/>')
        callouts.append(
            f'<text x="{bx:.1f}" y="{ty + (-4 if ty<by else 12):.1f}" font-size="10" '
            f'fill="var(--accent)" text-anchor="middle">bridge spine</text>')
    # densest node region -> "core" callout
    dens = _local_density(P, w, h)
    core = int(np.argmax(dens))
    cx0, cy0 = P[core]
    callouts.append(
        f'<text x="{cx0:.1f}" y="{cy0 + (18 if cy0 < h-40 else -10):.1f}" font-size="10" '
        f'fill="var(--ink-soft)" text-anchor="middle">mesh core</text>')

    # ---- caption + legend (mono) -------------------------------------------
    n_edges = len(a)
    caption = (
        f'<text x="{w-pad}" y="{h-12}" font-size="9" fill="var(--ink-faint)" '
        f'text-anchor="end">cam az 35 / el 22 deg · k={k} (cosine) · '
        f'{n_edges:,} edges · far edges first</text>')
    leg_block = (
        '<g font-family="ui-monospace,monospace" font-size="8.5">'
        f'<text x="{pad-4}" y="34" font-size="9" fill="var(--ink-soft)">'
        f'kNN mesh  k={k} (cosine)</text>'
        f'<line x1="{pad-4}" y1="48" x2="{pad+22}" y2="48" stroke="var(--ink-faint)" '
        f'stroke-width="0.75" stroke-opacity="0.40" vector-effect="non-scaling-stroke"/>'
        f'<text x="{pad+28}" y="51" fill="var(--ink-faint)">cosine edge — near brighter, far faded</text>'
        f'<line x1="{pad-4}" y1="62" x2="{pad+22}" y2="62" stroke="var(--accent)" '
        f'stroke-width="2.0" stroke-opacity="0.9" vector-effect="non-scaling-stroke"/>'
        f'<text x="{pad+28}" y="65" fill="var(--ink-soft)">bridge spine — load-bearing reach</text>'
        '</g>')

    body = (
        f'<g>{"".join(gn)}</g>'
        f'<g opacity="0.95">{"".join(edge_svg)}</g>'
        f'<g>{"".join(node_svg)}</g>'
        f'<g>{"".join(accent_svg)}</g>'
        f'<g>{"".join(end_svg)}</g>'
        f'<g font-size="10">{"".join(callouts)}{caption}</g>'
        f'{leg_block}'
    )

    return {
        "num": "3D 04", "order": 23, "name": "kNN mesh in 3-space", "tech": "knn · 3d",
        "why": (f"The k={k} neighbor graph drawn in the projected 3-space at the fixed "
                f"3D-01 camera; depth-faded hairline strands show where neighbors knot up, "
                f"and the accent spine marks the longest load-bearing reach across the cloud."),
        "svg": _svg(w, h,
                    f"kNN mesh of the embedding reservoir in projected 3-space at a fixed "
                    f"camera (azimuth 35 deg, elevation 22 deg). {n_edges:,} depth-graded "
                    f"k={k} cosine neighbor edges are drawn as hairline ink strands, far edges "
                    f"first so near strands overlay; dense regions read as edge-knots and the "
                    f"diffuse halo thins to single faint far edges. A single accent spine marks "
                    f"the longest load-bearing bridge edge spanning the cloud. Nodes are small "
                    f"depth-graded dots; a corner gnomon carries near/far depth ticks on the "
                    f"x/y/z axes and a mono legend distinguishes a faint cosine edge from the "
                    f"accent bridge.",
                    body),
        "legend": '<span><i class="f"></i> cosine edge (depth-faded)</span>'
                  '<span><i class="a"></i> bridge spine (load-bearing reach)</span>',
        "reveal": (f"<b>Reveals:</b> the manifold connectivity in 3-space — where neighbors "
                   f"knot into dense cores versus thin to single far edges, and the longest "
                   f"reach that ties distant regions together."),
        "cls": "",
    }
