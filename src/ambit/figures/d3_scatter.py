"""3D 01 — depth-cued projected 3D scatter.

The 3-D PCA reservoir (ctx.xyz) is projected through one fixed camera
(azimuth 35 deg, elevation 22 deg) to the 2-D page and painted strictly
back-to-front so occlusion is honest: nearer points are larger, fuller and
darker; farther points are smaller, fainter and lower-opacity. Density reads
purely by accumulation of faint translucent dots — no colour ramp across the
scatter. One accent spine carries the principal axis of the densest core. A
faint 3-arm graduated x/y/z gnomon anchors orientation; no bounding cube, no
floor grid.
"""

from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


def _camera(az_deg, el_deg):
    """Return (right, up, view) orthonormal basis for an az/el camera.

    `view` points from the scene toward the camera; a larger projection onto
    `view` means nearer to the viewer (used as the depth cue).
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    # camera direction (from origin toward eye)
    view = np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], float)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return right, up, view


@figure
def fig_d3_scatter(ctx):
    w, h, pad = 760, 470, 30

    xyz = np.asarray(ctx.xyz, float)
    # center so the camera basis spins about the cloud centroid
    c = xyz.mean(0)
    Q = xyz - c

    right, up, view = _camera(35.0, 22.0)

    # screen plane coords (u = right, v = up) and depth (toward viewer)
    u = Q @ right
    v = Q @ up
    depth = Q @ view  # larger = nearer the camera

    # map the projected plane into the svg box (y flipped, fit-to-width)
    P = _box(np.column_stack([u, v]), w, h, pad)

    # depth cue normalised 0..1 (1 = nearest)
    dlo, dhi = depth.min(), depth.max()
    dn = (depth - dlo) / max(dhi - dlo, 1e-9)

    # local accumulation density on the projected plane (drives accent core only)
    dens = _local_density(P, w, h)
    hot_thr = np.quantile(dens, 0.985)
    hot = dens >= hot_thr

    # paint back-to-front: farthest first so nearer points overdraw them
    order = np.argsort(depth)

    dots = []
    for i in order:
        d = dn[i]
        # near -> bigger + full var(--ink); far -> smaller + faint + translucent
        r = 0.9 + d * 1.5            # 0.9 .. 2.4
        op = 0.18 + d * 0.42         # 0.18 .. 0.60
        if d > 0.55:
            fill = "var(--ink)"
        elif d > 0.28:
            fill = "var(--ink-soft)"
        else:
            fill = "var(--ink-faint)"
        dots.append(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="{r:.2f}" '
                    f'fill="{fill}" fill-opacity="{op:.2f}"/>')

    # --- one accent spine: principal axis of the densest core, tilted into page
    spine = ""
    core_xyz = xyz[hot]
    if len(core_xyz) >= 3:
        cc = core_xyz.mean(0)
        # principal 3-D axis of the dense core
        C = core_xyz - cc
        cov = C.T @ C
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, -1]
        # extent of the core along that axis
        t = C @ axis
        half = 1.4 * float(np.abs(t).std() + 1e-6)
        a3 = cc + axis * half
        b3 = cc - axis * half
        # project the two endpoints through the same camera
        def proj(p3):
            q = p3 - c
            return np.array([q @ right, q @ up])
        ends = _box(np.vstack([proj(a3), proj(b3),
                               np.column_stack([u, v])]), w, h, pad)[:2]
        spine = (f'<line x1="{ends[0,0]:.1f}" y1="{ends[0,1]:.1f}" '
                 f'x2="{ends[1,0]:.1f}" y2="{ends[1,1]:.1f}" '
                 f'stroke="var(--accent)" stroke-width="2" stroke-linecap="round" '
                 f'opacity="0.92"/>')

    # --- faint 3-arm graduated gnomon (x/y/z), anchored lower-left of the cloud
    # build it in data space from the camera basis so it shares orientation.
    # pixels-per-unit estimate from the projected spread
    span_u = max(float(np.ptp(u)), 1e-9)
    px_per_u = (w - 2 * pad) / span_u  # approximate plane scale
    g_len_data = 0.55  # arm length in data units
    # origin of the gnomon in screen space (lower-left padding region)
    ox, oy = pad + 78, h - pad - 70
    axes_world = [np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0])]
    labels = ["x", "y", "z"]
    gnomon = ['<g class="gnomon">']
    arm_px = g_len_data * px_per_u
    for ax, lab in zip(axes_world, labels):
        su = ax @ right
        sv = ax @ up
        ex = ox + su * arm_px
        ey = oy - sv * arm_px
        gnomon.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{ex:.2f}" y2="{ey:.2f}" '
                      f'stroke="var(--ink-faint)" stroke-width="0.6" '
                      f'stroke-linecap="round" opacity="0.7"/>')
        # graduated reference ticks along the arm (0.25,0.5,0.75,1.0 of length)
        perp = np.array([-sv, su])
        plen = np.hypot(*perp)
        if plen > 1e-6:
            perp = perp / plen
        for frac in (0.25, 0.5, 0.75, 1.0):
            tx = ox + su * arm_px * frac
            ty = oy - sv * arm_px * frac
            tl = 4.0 if frac == 1.0 else 2.6
            op = 0.6 if frac == 1.0 else 0.4
            gnomon.append(
                f'<line x1="{tx - perp[0]*tl:.2f}" y1="{ty - perp[1]*tl:.2f}" '
                f'x2="{tx + perp[0]*tl:.2f}" y2="{ty + perp[1]*tl:.2f}" '
                f'stroke="var(--rule-soft)" stroke-width="0.5" '
                f'stroke-linecap="round" opacity="{op}"/>')
            if frac in (0.5, 1.0):
                gnomon.append(
                    f'<text x="{tx + perp[0]*7:.2f}" y="{ty + perp[1]*7 + 2:.2f}" '
                    f'font-size="6.5" font-family="ui-monospace, monospace" '
                    f'fill="var(--ink-faint)" opacity="0.55" '
                    f'text-anchor="middle">{frac:.1f}</text>')
        # axis label at the arm tip
        lx = ox + su * arm_px * 1.16
        ly = oy - sv * arm_px * 1.16
        gnomon.append(f'<text x="{lx:.2f}" y="{ly + 3:.2f}" font-size="9" '
                      f'fill="var(--ink-faint)" opacity="0.85" '
                      f'text-anchor="middle">{lab}</text>')
    gnomon.append('</g>')

    erank = metrics.effective_rank(ctx.eigs)
    thin = float(xyz.std(0)[2] / xyz.std(0)[0])  # z spread vs x spread

    body = "".join(dots) + spine + "".join(gnomon)
    aria = ("Depth-cued projected 3D scatter of the embedding reservoir at one "
            "fixed camera (azimuth 35 degrees, elevation 22 degrees); points "
            "painted back-to-front, nearer larger and darker, farther smaller "
            "and fainter; one accent spine on the densest core, a faint 3-arm "
            "graduated x/y/z gnomon, no bounding cube.")

    return {
        "num": "3D 01", "order": 20, "name": "Depth-cued 3D scatter", "tech": "3d · projection",
        "why": (f"The 3-D PCA reservoir seen through one fixed camera and painted "
                f"back-to-front: near points larger and inked, far points small and faint, "
                f"so occlusion and depth read honestly. The cloud is broad in x-y but thin in z "
                f"(z spread ≈ {thin:.2f}× x)."),
        "svg": _svg(w, h, aria, body),
        "legend": '<span><i class="f"></i> point — density by accumulation</span>'
                  '<span><i class="f"></i> near / far depth cue (size · ink · opacity)</span>'
                  '<span><i class="a"></i> dense-core principal axis (accent spine)</span>',
        "reveal": (f"<b>Reveals:</b> the genuine 3-D shape of the projected cloud — a flattened "
                   f"slab (effective rank ≈ {erank:.0f}) with its dense core's principal axis "
                   f"tilted into the page."),
        "cls": "",
    }
