"""Localized anisotropy — the unsupervised, multiscale local-concentration field.

Crowding is *local*: a dataset can be globally isotropic while a sub-population is
crammed into a cone, and a global scalar is non-collapsible over clusters, so it
hides or misreads the pocket. (See docs/concepts/cluster-sensitive-anisotropy.html.)

We measure, per item, how concentrated its k-nearest-neighbor neighborhood is — at
several scales k — and read the DISTRIBUTION of that field:

  - unimodal at the bulk            -> roomy / uniform
  - the bulk shifted far up          -> globally crowded (the "dandelion")
  - a separated high mode            -> a crowded pocket

Fully unsupervised. Calibrated internally for *local* pockets (the field vs its own
median/MAD) and against a synthetic *isotropic reference* for *global* crowding.
Labels are never required; when present they add a neighborhood-relevance overlay
(handled by the caller, not here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import metrics

DEFAULT_SCALES = (10, 50, 200)


@dataclass
class Pocket:
    members: np.ndarray         # reservoir indices in this crowded pocket
    size: int
    concentration: float        # mean local concentration (mean cosine to k-NN)
    margin: float               # mean NN cosine margin (top1 - top2) -> resolution
    isoscore_star: float        # collapsed sliver (low) vs round dense ball (high), n<d-robust
    scale: int                  # k at which the pocket is most anomalous
    z: float                    # peak crowding score in the pocket


@dataclass
class LocalAnisotropy:
    field: np.ndarray           # (m,) per-item concentration at the headline scale
    score: np.ndarray           # (m,) signed multiscale crowding z (robust)
    scale_star: int             # headline scale (most separated field)
    scales: tuple
    per_scale: dict             # k -> (m,) concentration
    iso_ref: dict               # k -> (n_ref,) isotropic-reference concentration
    bulk: float                 # dataset field median at scale_star
    iso_bulk: float             # isotropic-reference median at scale_star
    global_crowding: float      # (bulk - iso_bulk)/iso_mad : whole-space crowding
    multimodal: bool            # a high mode separated from the bulk?
    valley: Optional[float]     # field value of the bulk/pocket split (if multimodal)
    crowded_fraction: float     # fraction of items flagged crowded
    pockets: list               # list[Pocket], crowded -> roomy order


def for_ctx(ctx, **kw):
    """Compute (and memoize on the report Ctx) the localized-anisotropy result.

    RES 06 (the field histogram) and RES 07 (the crowding map) read the *same*
    object, so it is computed once on first access and cached on the Ctx. All the
    numbers come from this generation step — the figures only draw them."""
    cached = getattr(ctx, "_local_aniso", None)
    if cached is None:
        cached = localized_anisotropy(ctx.es.X, labels=getattr(ctx, "labels", None), **kw)
        try:
            ctx._local_aniso = cached
        except Exception:
            pass
    return cached


def _topk_sims(X: np.ndarray, kmax: int, block: int = 2048):
    """For each row, the top-`kmax` cosine similarities (descending, self excluded)
    and their column indices. Blocked exact kNN — O(m^2 d), fine on the reservoir."""
    m = X.shape[0]
    kmax = int(min(kmax, m - 1))
    sims = np.empty((m, kmax), np.float32)
    idx = np.empty((m, kmax), np.int32)
    Xt = np.ascontiguousarray(X.T)
    for s in range(0, m, block):
        e = min(s + block, m)
        S = (X[s:e] @ Xt).astype(np.float32, copy=False)
        S[np.arange(e - s), s + np.arange(e - s)] = -2.0          # drop self
        part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
        rows = np.arange(e - s)[:, None]
        vals = S[rows, part]
        order = np.argsort(-vals, axis=1)
        sims[s:e] = vals[rows, order]
        idx[s:e] = part[rows, order]
    return sims, idx


def _mad(a):
    med = float(np.median(a))
    return med, float(1.4826 * np.median(np.abs(a - med)))


def _angular_components(X, dist_thr):
    """Split a set of (crowded) unit vectors into pockets by angular proximity. The
    threshold sits between within-pocket and between-pocket cosine distance, so a
    single tight cone stays one pocket and distinct cones separate (unlike a k-means
    with a floor of >=3 clusters, which over-segments)."""
    n = len(X)
    if n <= 6000:
        try:
            from sklearn.cluster import AgglomerativeClustering
            ac = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                         linkage="average", distance_threshold=dist_thr)
            return ac.fit_predict(X)
        except Exception:
            pass
    # greedy fallback (memory-safe, order-dependent): grow pocket seeds
    cen = []
    lab = np.full(n, -1, int)
    c0 = 1.0 - dist_thr
    for i in range(n):
        best, bestc = -1, c0
        for j, c in enumerate(cen):
            s = float(X[i] @ c)
            if s > bestc:
                bestc, best = s, j
        if best < 0:
            cen.append(X[i].copy()); lab[i] = len(cen) - 1
        else:
            lab[i] = best
            cen[best] = cen[best] + (X[i] - cen[best]) / 8.0     # nudge the seed
    return lab


def _multimodal(field, frac_min: float = 0.004):
    """A high mode separated from the bulk by a near-empty valley? Returns
    (is_multimodal, valley_value)."""
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-6:
        return False, None
    nb = 60
    counts, edges = np.histogram(field, bins=nb, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = float(np.median(field))
    bb = max(0, min(nb - 1, int((med - lo) / (hi - lo) * nb)))
    thr = max(1.0, counts.max() * 0.02)
    i = bb
    while i < nb and counts[i] > thr:                              # walk off the bulk
        i += 1
    vstart = i
    while i < nb and counts[i] <= thr:                             # cross the valley
        i += 1
    if vstart < i < nb and counts[i:].sum() >= field.size * frac_min:
        return True, float(centers[(vstart + i) // 2])
    return False, None


def localized_anisotropy(X, *, labels: Optional[np.ndarray] = None,
                         scales=DEFAULT_SCALES, threshold: float = 3.0,
                         n_ref: int = 4000, seed: int = 0) -> LocalAnisotropy:
    X = np.asarray(X, np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    m, d = X.shape
    scales = tuple(int(k) for k in scales if 2 <= k < m)
    if not scales:
        scales = (max(2, m // 4),)
    kmax = max(scales)

    sims, idx = _topk_sims(X, kmax)
    per_scale = {k: sims[:, :k].mean(1) for k in scales}

    # isotropic reference: a uniform same-density sample (k scaled to hold k/N fixed)
    rng = np.random.default_rng(seed)
    nref = int(min(n_ref, m))
    R = rng.standard_normal((nref, d)).astype(np.float32)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    kref = {k: max(2, int(round(k * nref / m))) for k in scales}
    rsims, _ = _topk_sims(R, max(kref.values()))
    iso_ref = {k: rsims[:, :kref[k]].mean(1) for k in scales}

    # internal robust z per scale, multiscale score = most extreme |z| (signed)
    zs = {}
    for k in scales:
        med, mad = _mad(per_scale[k])
        zs[k] = (per_scale[k] - med) / (mad or 1e-9)
    Z = np.stack([zs[k] for k in scales], 1)
    ai = np.argmax(np.abs(Z), 1)
    score = Z[np.arange(m), ai]
    best_k = np.array(scales)[ai]

    # headline scale = the one whose field has the most extended upper tail (MAD units)
    def sep(k):
        med, mad = _mad(per_scale[k])
        return (np.quantile(per_scale[k], 0.99) - med) / (mad or 1e-9)
    scale_star = max(scales, key=sep)
    field = per_scale[scale_star]
    bulk = float(np.median(field))
    iso_med, iso_mad = _mad(iso_ref[scale_star])
    global_crowding = float((bulk - iso_med) / (iso_mad or 1e-9))

    multimodal, valley = _multimodal(field)

    # crowded items = the separated high mode (above the valley). Tying pocket
    # membership to the distribution avoids the per-item tail over-flagging that a
    # z-threshold suffers at large m across several scales; uniform data (no valley)
    # yields no pockets. `score` (multiscale z) is still returned for map coloring.
    if multimodal and valley is not None:
        crowded = field > valley
    else:
        crowded = np.zeros(m, bool)
    crowded_fraction = float(crowded.mean())

    pockets = []
    cr = np.where(crowded)[0]
    if cr.size >= max(8, int(m * 0.002)):
        thr_d = float(np.clip(1.0 - 0.5 * field[cr].mean(), 0.35, 0.95))
        plabs = _angular_components(X[cr], thr_d)
        for lab in sorted(set(int(x) for x in plabs)):
            if lab < 0:
                continue
            mem = cr[plabs == lab]
            if mem.size < max(5, int(m * 0.001)):
                continue
            cov = np.cov(X[mem].T) if mem.size > 1 else np.eye(d)
            iss = metrics.isoscore_star(metrics.eigs_from_cov(cov), int(mem.size))
            sub = mem[np.argmax(np.abs(score[mem]))]
            pockets.append(Pocket(
                members=mem, size=int(mem.size),
                concentration=float(field[mem].mean()),
                margin=float(np.mean(sims[mem, 0] - sims[mem, 1])),
                isoscore_star=float(iss), scale=int(best_k[sub]),
                z=float(score[mem].max())))
        pockets.sort(key=lambda p: -p.z)

    return LocalAnisotropy(field=field, score=score, scale_star=int(scale_star),
                           scales=scales, per_scale=per_scale, iso_ref=iso_ref,
                           bulk=bulk, iso_bulk=float(iso_med),
                           global_crowding=global_crowding, multimodal=bool(multimodal),
                           valley=(float(valley) if valley is not None else None),
                           crowded_fraction=crowded_fraction, pockets=pockets)
