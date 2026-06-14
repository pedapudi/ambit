"""Synthetic validation for localized_anisotropy: each case has a known right answer
for the field shape. Run: PYTHONPATH=src .venv/bin/python scripts/validate_local_anisotropy.py
"""
import numpy as np

from ambit.local_anisotropy import localized_anisotropy

rng = np.random.default_rng(0)
D = 768


def sphere(n):
    x = rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cone(n, kappa):
    u = rng.standard_normal(D); u /= np.linalg.norm(u)
    x = kappa * u + rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


cases = {
    "uniform":               sphere(9000),
    "background + 2 pockets": np.vstack([sphere(8000), cone(500, 55), cone(500, 55)]),
    "dandelion (40 cones)":  np.vstack([cone(225, 35) for _ in range(40)]),
    "background + 1 pocket":  np.vstack([sphere(8500), cone(500, 55)]),
}

# expectations: (multimodal, n_pockets, global_crowding 'high'?)
expect = {
    "uniform":               (False, 0, False),
    "background + 2 pockets": (True,  2, False),
    "dandelion (40 cones)":  (False, 0, True),
    "background + 1 pocket":  (True,  1, False),
}

print(f"{'case':24s} {'multimodal':>10} {'pockets':>8} {'crowded%':>9} {'global':>8} {'scale*':>7}  verdict")
allok = True
for name, X in cases.items():
    r = localized_anisotropy(X)
    em, ep, eg = expect[name]
    ok = (r.multimodal == em) and (len(r.pockets) == ep) and ((r.global_crowding > 8) == eg)
    allok = allok and ok
    print(f"{name:24s} {str(r.multimodal):>10} {len(r.pockets):>8} {r.crowded_fraction*100:>8.1f}% "
          f"{r.global_crowding:>8.1f} {r.scale_star:>7}  {'ok' if ok else 'FAIL'}")
    for p in r.pockets[:3]:
        print(f"    pocket n={p.size:5d} conc={p.concentration:.2f} margin={p.margin:.3f} "
              f"iso*={p.isoscore_star:.2f} z={p.z:.1f} k={p.scale}")

print("\nALL PASS" if allok else "\nSOME FAILED")
