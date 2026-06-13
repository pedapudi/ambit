"""3D 03 — Isometric voxel occupancy lattice.

Bin ``ctx.xyz`` into a coarse 9x9x9 cubic lattice and draw the occupied voxels
as stepped isometric cubes, painted back-to-front, tinted by binned point count
via an accent->bad volume heatmap. The dataset's cavity reads as the run of
*missing* voxels punched through the interior — ringed by a dashed zero-density
mouth. Depth comes from iso face-shading only; no bounding cube, no floor grid.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


# isometric camera (az35 / el22-ish): unit lattice steps -> screen deltas
_AX = 26.0   # half-width of a voxel face along x screen axis
_AY = 15.0   # vertical shear of x/y steps
_AZ = 30.0   # vertical rise of a z step


def _iso(i, j, k, ox, oy):
    """Project lattice index (i,j,k) to screen (px,py) at the voxel's near-top
    corner. +i goes down-right, +j goes down-left, +k goes up."""
    px = ox + (i - j) * _AX
    py = oy + (i + j) * _AY - k * _AZ
    return px, py


def _cube_faces(i, j, k, ox, oy, fill):
    """Three visible iso faces (top, left, right) of voxel (i,j,k), with --ink
    face-shading overlays so depth reads without any outline cube."""
    # eight relevant corners as (px,py)
    p000 = _iso(i,   j,   k,   ox, oy)
    p100 = _iso(i+1, j,   k,   ox, oy)
    p010 = _iso(i,   j+1, k,   ox, oy)
    p110 = _iso(i+1, j+1, k,   ox, oy)
    p001 = _iso(i,   j,   k+1, ox, oy)
    p101 = _iso(i+1, j,   k+1, ox, oy)
    p011 = _iso(i,   j+1, k+1, ox, oy)
    p111 = _iso(i+1, j+1, k+1, ox, oy)

    def poly(pts, f, op=None, stroke=True):
        s = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        if op is None:
            return (f'<polygon points="{s}" fill="{f}" stroke="var(--rule)" '
                    f'stroke-width="0.6" stroke-linejoin="round"/>')
        return f'<polygon points="{s}" fill="{f}" opacity="{op}" stroke="none"/>'

    out = []
    # top face (k+1): brightest, no shade overlay
    out.append(poly([p001, p101, p111, p011], fill))
    # left face (j+1 side): mid shade
    out.append(poly([p010, p110, p111, p011], fill))
    out.append(poly([p010, p110, p111, p011], "var(--ink)", op="0.30"))
    # right face (i+1 side): light shade
    out.append(poly([p100, p110, p111, p101], fill))
    out.append(poly([p100, p110, p111, p101], "var(--ink)", op="0.16"))
    return out


def _largest_empty_run(occ):
    """Find the longest contiguous run of EMPTY voxels through the interior of
    the lattice (axis-aligned, any of 3 directions). Returns the run's center
    (i,j,k) in lattice coords and its length, or None if the lattice is solid."""
    g = occ.shape[0]
    best = None  # (length, center_idx tuple)
    for axis in range(3):
        for a in range(g):
            for b in range(g):
                # walk the line along `axis`
                run = 0
                start = 0
                for c in range(g + 1):
                    inside = c < g
                    if inside:
                        idx = [0, 0, 0]
                        idx[axis] = c
                        rem = [x for x in range(3) if x != axis]
                        idx[rem[0]] = a
                        idx[rem[1]] = b
                        empty = not occ[idx[0], idx[1], idx[2]]
                    else:
                        empty = False
                    if inside and empty:
                        if run == 0:
                            start = c
                        run += 1
                    else:
                        # run ended at [start, c)
                        # require it to be *interior* (not touching a face on
                        # either end) so it reads as a punched cavity
                        if run >= 2 and start > 0 and c < g:
                            if best is None or run > best[0]:
                                cmid = (start + c - 1) / 2.0
                                center = [0, 0, 0]
                                center[axis] = cmid
                                rem = [x for x in range(3) if x != axis]
                                center[rem[0]] = a
                                center[rem[1]] = b
                                best = (run, tuple(center))
                        run = 0
    return best


@figure
def fig_d3_voxel(ctx):
    W, H = 820, 620
    G = 9  # 9^3 lattice

    xyz = None if getattr(ctx, "xyz", None) is None else np.asarray(ctx.xyz, float)
    if xyz is None or xyz.shape[0] == 0:
        note = ('<text x="%d" y="%d" font-size="12" fill="var(--ink-faint)" '
                'text-anchor="middle">needs 3-D projection (ctx.xyz)</text>'
                % (W // 2, H // 2))
        return {
            "num": "3D 03", "order": 22, "name": "Isometric voxel occupancy",
            "tech": "voxels",
            "why": "Bins the 3-D projection into a coarse cubic lattice; "
                   "occupied voxels read as an isometric volume, the cavity as missing cells.",
            "svg": _svg(W, H, "Isometric voxel occupancy — 3-D projection unavailable", note),
            "legend": '<span><i class="f"></i> needs ctx.xyz</span>',
            "reveal": "<b>Reveals:</b> the volumetric occupancy of the embedding hull.",
            "cls": "",
        }

    # --- bin into the lattice ---------------------------------------------
    mn = xyz.min(0)
    span = np.maximum(xyz.max(0) - mn, 1e-9)
    ijk = np.clip(((xyz - mn) / span * G).astype(int), 0, G - 1)
    counts = np.zeros((G, G, G), dtype=int)
    np.add.at(counts, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)
    occ = counts > 0
    n_occ = int(occ.sum())
    cmax = int(counts.max()) or 1
    occ_frac = n_occ / float(G ** 3)

    # hottest voxel (megacluster core)
    fi, fj, fk = np.unravel_index(int(np.argmax(counts)), counts.shape)
    core_n = int(counts[fi, fj, fk])

    # interior cavity = longest empty run
    cav = _largest_empty_run(occ)

    # --- iso layout: center the lattice in the box ------------------------
    # lattice screen extent: i,j in [0,G], k in [0,G]
    # x spread = (i-j) in [-G, G] -> width 2*G*_AX ; vertical from (i+j) and k.
    cx = W / 2.0
    # place so the whole cube sits comfortably with room for legend at bottom.
    ox = cx
    oy = 250.0  # screen origin for lattice index (0,0,0) near-top corner

    body = []

    # ---- light graduated iso gnomon (x/y/z axes with ticks) --------------
    # gnomon origin in lower-left, away from the lattice mass
    gx0, gy0 = 118.0, 458.0
    glen = 3  # axis length in lattice steps
    # x axis: +i  (down-right)
    xe = (gx0 + glen * _AX, gy0 + glen * _AY)
    # y axis: +j  (down-left)
    ye = (gx0 - glen * _AX, gy0 + glen * _AY)
    # z axis: +k  (straight up)
    ze = (gx0, gy0 - glen * _AZ)
    body.append('<g stroke="var(--ink-faint)" stroke-width="0.6" fill="none" opacity="0.7">')
    body.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{xe[0]:.1f}" y2="{xe[1]:.1f}"/>')
    body.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{ye[0]:.1f}" y2="{ye[1]:.1f}"/>')
    body.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{ze[0]:.1f}" y2="{ze[1]:.1f}"/>')
    for t in range(1, glen + 1):
        # tick marks along each axis
        body.append(f'<line x1="{gx0+t*_AX:.1f}" y1="{gy0+t*_AY:.1f}" '
                    f'x2="{gx0+t*_AX:.1f}" y2="{gy0+t*_AY+5:.1f}"/>')
        body.append(f'<line x1="{gx0-t*_AX:.1f}" y1="{gy0+t*_AY:.1f}" '
                    f'x2="{gx0-t*_AX:.1f}" y2="{gy0+t*_AY+5:.1f}"/>')
        body.append(f'<line x1="{gx0:.1f}" y1="{gy0-t*_AZ:.1f}" '
                    f'x2="{gx0-5:.1f}" y2="{gy0-t*_AZ:.1f}"/>')
    body.append('</g>')
    body.append('<g fill="var(--ink-faint)" font-size="9" opacity="0.9" '
                'font-family="ui-monospace, monospace">')
    body.append(f'<text x="{xe[0]+6:.1f}" y="{xe[1]+10:.1f}">x</text>')
    body.append(f'<text x="{ye[0]-12:.1f}" y="{ye[1]+10:.1f}">y</text>')
    body.append(f'<text x="{ze[0]-3:.1f}" y="{ze[1]-6:.1f}">z</text>')
    body.append('</g>')

    # ---- occupied voxels, painted back-to-front --------------------------
    # painter's order: smaller (i + j - k) is farther back; draw ascending.
    cells = []
    for i in range(G):
        for j in range(G):
            for k in range(G):
                if occ[i, j, k]:
                    depth = i + j - k
                    cells.append((depth, i, j, k))
    cells.sort(key=lambda c: c[0])

    for depth, i, j, k in cells:
        c = counts[i, j, k]
        frac = (c - 1) / max(1, cmax - 1)  # 0..1 over the occupied range
        pct = int(round(frac * 92))        # 0% accent .. 92% bad (ramp cap)
        fill = f"color-mix(in srgb, var(--bad) {pct}%, var(--accent))"
        body.extend(_cube_faces(i, j, k, ox, oy, fill))
        if (i, j, k) == (fi, fj, fk):
            # accent core outline on the megacluster's hottest voxel
            p001 = _iso(i, j, k + 1, ox, oy)
            p101 = _iso(i + 1, j, k + 1, ox, oy)
            p111 = _iso(i + 1, j + 1, k + 1, ox, oy)
            p011 = _iso(i, j + 1, k + 1, ox, oy)
            top = f"{p001[0]:.1f},{p001[1]:.1f} {p101[0]:.1f},{p101[1]:.1f} " \
                  f"{p111[0]:.1f},{p111[1]:.1f} {p011[0]:.1f},{p011[1]:.1f}"
            body.append(f'<polygon points="{top}" fill="none" '
                        f'stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>')

    # core call-out label
    core_px, core_py = _iso(fi + 0.5, fj + 0.5, fk + 1, ox, oy)
    body.append(f'<line x1="{core_px:.1f}" y1="{core_py:.1f}" '
                f'x2="{core_px+34:.1f}" y2="{core_py-26:.1f}" '
                f'stroke="var(--accent)" stroke-width="1" opacity="0.7"/>')
    body.append(f'<text x="{core_px+38:.1f}" y="{core_py-26:.1f}" fill="var(--accent)" '
                f'font-size="10">core · {core_n} pts</text>')

    # ---- cavity: dashed zero-density mouth on the missing run ------------
    if cav is not None:
        _, (ci, cj, ck) = cav
        vpx, vpy = _iso(ci + 0.5, cj + 0.5, ck + 0.5, ox, oy)
        body.append(f'<ellipse cx="{vpx:.1f}" cy="{vpy:.1f}" rx="32" ry="16" '
                    f'fill="none" stroke="var(--ink-faint)" stroke-width="1" '
                    f'stroke-dasharray="4 4"/>')
        body.append(f'<text x="{vpx:.1f}" y="{vpy+4:.1f}" fill="var(--ink-faint)" '
                    f'font-size="10" text-anchor="middle">void</text>')

    # ---- title strip -----------------------------------------------------
    body.append(f'<text x="20" y="28" fill="var(--ink-soft)" font-size="11" '
                f'font-family="ui-monospace, monospace">{ctx.scan.source} · '
                f'{G}×{G}×{G} voxel lattice · {n_occ}/{G**3} cells occupied '
                f'({occ_frac*100:.0f}%)</text>')

    # ---- count-ramp legend bar (accent cool -> bad hot) ------------------
    lx0, lx1, ly = 556.0, 788.0, 568.0
    lw = lx1 - lx0
    body.append(f'<text x="{lx0:.0f}" y="{ly-12:.0f}" font-size="10" '
                f'fill="var(--ink-soft)">binned point count / voxel</text>')
    # smooth ramp as stepped color-mix segments (accent cool -> bad hot),
    # no gradient def / url() reference so the svg stays fully tokenized.
    nseg = 48
    sw = lw / nseg
    for s in range(nseg):
        pct = int(round(s / (nseg - 1) * 92))
        body.append(f'<rect x="{lx0+s*sw:.2f}" y="{ly:.0f}" width="{sw+0.6:.2f}" '
                    f'height="12" fill="color-mix(in srgb, var(--bad) {pct}%, '
                    f'var(--accent))" stroke="none"/>')
    body.append(f'<rect x="{lx0:.0f}" y="{ly:.0f}" width="{lw:.0f}" height="12" '
                f'fill="none" stroke="var(--rule)" stroke-width="0.6"/>')
    # labeled breakpoints with hairline ticks
    qmid = max(1, cmax // 2)
    qq = max(1, cmax // 4)
    bps = [(0, 0), (qq, 0.25), (qmid, 0.5), (cmax, 1.0)]
    anchors = ["start", "middle", "middle", "end"]
    for (val, fr), anc in zip(bps, anchors):
        tx = lx0 + fr * lw
        body.append(f'<line x1="{tx:.1f}" y1="{ly+12:.0f}" x2="{tx:.1f}" '
                    f'y2="{ly+17:.0f}" stroke="var(--rule-soft)"/>')
        body.append(f'<text x="{tx:.1f}" y="{ly+28:.0f}" font-size="9" '
                    f'fill="var(--ink-faint)" text-anchor="{anc}" '
                    f'font-family="ui-monospace, monospace">{val}</text>')

    aria = (f"Isometric voxel-occupancy lattice of {ctx.scan.source} binned into a "
            f"{G} by {G} by {G} cubic grid; only occupied voxels are drawn as stepped "
            f"iso cubes tinted by binned point count. Occupied voxels fill about "
            f"{occ_frac*100:.0f} percent of the hull volume; the hottest cell carrying "
            f"the accent outline is the megacluster core at {core_n} points; runs of "
            f"empty cells read as the missing interior cavity, ringed by a dashed "
            f"zero-density mouth. Depth from iso face-shading only, no bounding cube.")

    return {
        "num": "3D 03", "order": 22, "name": "Isometric voxel occupancy",
        "tech": "voxels",
        "why": (f"The 3-D projection binned into a {G}×{G}×{G} lattice; occupied "
                f"voxels are stacked isometrically and tinted by point count, so the "
                f"filled volume and the punched-through cavity read at a glance."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="a"></i> low count (accent) → '
                  '<i class="b"></i> high count (bad)</span>'
                  '<span><i class="dash"></i> zero-density cavity mouth</span>'
                  '<span><i class="a"></i> megacluster core (accent outline)</span>',
        "reveal": (f"<b>Reveals:</b> the volumetric occupancy of the embedding hull — "
                   f"only {occ_frac*100:.0f}% of the {G**3} lattice cells hold any mass, "
                   f"and the empty interior cells expose the dataset's cavity."),
        "cls": "",
    }
