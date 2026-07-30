"""Per-entity crowding — the distance-to-measure field and the merge tree.

The local half of docs/concepts/continuous-occupancy.md:

- **DTM field** (Chazal, Cohen-Steiner & Mérigot, 2011): each entity's score is the
  root-mean-square distance to its nearest ⌈m_frac·n⌉ reservoir points — "the radius
  of the ball this entity needs to gather a fixed share of the corpus." Small =
  crowded, large = isolated; the low tail names the crowded entities, the high tail
  the voids. Uniformly Lipschitz in the Wasserstein distance of the input (the
  stability guarantee bin counts lack); the one knob is a mass *fraction*, not a
  length, bin width, or k.

- **Merge tree / pockets** (single linkage on the mutual-reachability metric — the
  HDBSCAN construction; Campello et al. 2013/2015; equivalently H0 persistence of
  the density filtration): connect entities by shortest bridges and watch groups
  merge as the scale grows. Each tight group gets a birth scale, a death (merge into
  an older group, by the elder rule), members, and a prominence — with no flat
  threshold chosen anywhere. Answers: how many over-tight pockets, how tight, which
  entities, and at what scale each detaches from the bulk.

Pure numpy; O(n²) done blockwise at reservoir scale, subsampled above `max_n`.
Distances here are cosine distances (1 − cos), matching the kNN layer's convention.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- DTM field
def topk_cos_values(Xn, k: int, block: int = 2048) -> np.ndarray:
    """Top-k cosine similarity *values* per row (self excluded), descending, blocked
    so the full Gram is never materialized. `Xn` must be L2-normalized."""
    Xn = np.ascontiguousarray(Xn, np.float32)
    m = len(Xn)
    k = int(min(k, m - 1))
    out = np.empty((m, max(k, 1)), np.float32)
    Xt = np.ascontiguousarray(Xn.T)
    for s in range(0, m, block):
        e = min(s + block, m)
        S = Xn[s:e] @ Xt
        S[np.arange(e - s), s + np.arange(e - s)] = -2.0          # drop self
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        vals = S[rows, part]
        out[s:e, :k] = -np.sort(-vals, axis=1)
    return out[:, :k]


def dtm(Xn, m_frac: float = 0.02, k: int = None, block: int = 2048) -> np.ndarray:
    """Per-entity distance-to-measure over the reservoir: √(mean of the squared chord
    distances to the nearest k = ⌈m_frac·n⌉ points). Returns (n,) float64."""
    Xn = np.asarray(Xn)
    n = len(Xn)
    if k is None:
        k = max(2, int(np.ceil(m_frac * n)))
    k = int(min(k, n - 1))
    c = topk_cos_values(Xn, k, block=block).astype(np.float64)
    return np.sqrt(np.clip(2.0 - 2.0 * c, 0.0, None).mean(1))


def dtm_null_band(n: int, dim: int, m_frac: float = 0.02, seed: int = 0,
                  sample: int = 2048):
    """(lo, hi): the 1st/99th percentile of the DTM field for a uniform corpus of
    matched size and dimension — the reference band the data field is read against.
    Computed on a capped subsample with the same mass fraction (DTM's knob is a
    fraction, so the band transfers)."""
    rng = np.random.default_rng(seed)
    s = int(min(n, sample))
    X = rng.standard_normal((s, dim)).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    f = dtm(X, m_frac=m_frac)
    return float(np.percentile(f, 1)), float(np.percentile(f, 99))


# ---------------------------------------------------------------- merge tree
def _mst_prim(D: np.ndarray):
    """MST of a dense symmetric distance matrix via Prim; returns (weight, i, j)
    edges. O(n²) with vectorized relaxation — fine at merge-tree scale."""
    n = len(D)
    in_tree = np.zeros(n, bool)
    best = np.full(n, np.inf)
    best_from = np.zeros(n, np.int64)
    in_tree[0] = True
    np.minimum(best, D[0], out=best)
    best_from[:] = 0
    best[0] = np.inf
    edges = []
    for _ in range(n - 1):
        v = int(np.argmin(best))
        edges.append((float(best[v]), int(best_from[v]), v))
        in_tree[v] = True
        best[v] = np.inf
        row = D[v]
        upd = (~in_tree) & (row < best)
        best[upd] = row[upd]
        best_from[upd] = v
    return edges


def spanning_edges(Xn, *, max_n: int = 4096, seed: int = 0):
    """(edges, index_map): the minimum spanning tree of the reservoir under plain
    cosine distance — the corpus's shortest bridges, for display. Uses the same
    seeded subsample as `pockets` (same seed ⇒ same rows), so pocket membership
    and MST vertices align. Edges are (weight, i, j) with i/j indexing the
    subsample; map to reservoir rows via index_map."""
    Xn = np.ascontiguousarray(np.asarray(Xn), np.float32)
    n0 = len(Xn)
    rng = np.random.default_rng(seed)
    idx_map = np.arange(n0)
    if n0 > max_n:
        idx_map = rng.choice(n0, max_n, replace=False)
        Xn = Xn[idx_map]
    D = np.clip(1.0 - (Xn @ Xn.T).astype(np.float64), 0.0, None)
    np.fill_diagonal(D, 0.0)
    return _mst_prim(D), idx_map


def pockets(Xn, *, min_size: int = 8, k_core: int = None, max_n: int = 4096,
            max_pockets: int = 12, seed: int = 0):
    """Prominence-ranked tight pockets from the condensed merge tree.

    Single linkage on the mutual-reachability distance (cosine distance vs the
    k_core-th neighbor floor — the HDBSCAN construction) gives the dendrogram; it
    is condensed with `min_size` (a split is real only when both sides are at
    least `min_size`; smaller side-branches fall out of their cluster), and
    clusters are selected by **excess of mass** in λ = 1/distance, so a tight
    pocket that persists over a long scale range beats both its brief parent and
    its splintered children. Each selected pocket reports: `size`, `birth` (the
    scale at which its first `min_size` members hold together), `death` (the
    scale at which it merges with its sibling), `prominence` = death − birth,
    and `members` (indices into the original reservoir via the returned
    index_map). No flat threshold is chosen anywhere."""
    Xn = np.ascontiguousarray(np.asarray(Xn), np.float32)
    n0 = len(Xn)
    rng = np.random.default_rng(seed)
    idx_map = np.arange(n0)
    if n0 > max_n:
        idx_map = rng.choice(n0, max_n, replace=False)
        Xn = Xn[idx_map]
    n = len(Xn)
    if n < 2 * min_size:
        return [], idx_map
    G = Xn @ Xn.T
    D = np.clip(1.0 - G.astype(np.float64), 0.0, None)            # cosine distance
    np.fill_diagonal(D, 0.0)
    if k_core is None:
        k_core = max(5, n // 256)
    k_core = int(min(k_core, n - 1))
    core = np.partition(D + np.where(np.eye(n, dtype=bool), np.inf, 0.0),
                        k_core - 1, axis=1)[:, k_core - 1]
    MR = np.maximum(D, np.maximum(core[:, None], core[None, :]))  # mutual reachability
    edges = sorted(_mst_prim(MR))

    # ---- dendrogram: nodes 0..n-1 leaves; each merge appends a node ---------
    m_nodes = 2 * n - 1
    left = np.full(m_nodes, -1, np.int64)
    right = np.full(m_nodes, -1, np.int64)
    weight = np.zeros(m_nodes)
    count = np.ones(m_nodes, np.int64)
    first_w = np.full(m_nodes, np.inf)         # min scale at which a >=min_size sub-comp exists
    parent = np.arange(n)
    comp_node = np.arange(n)                   # union-find root -> dendrogram node

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = n
    for w, i, j in edges:
        ra, rb = find(i), find(j)
        if ra == rb:
            continue
        na, nb = comp_node[ra], comp_node[rb]
        left[nxt], right[nxt], weight[nxt] = na, nb, w
        count[nxt] = count[na] + count[nb]
        fw = min(first_w[na], first_w[nb])
        if count[nxt] >= min_size and not np.isfinite(fw):
            fw = w
        first_w[nxt] = fw
        parent[rb] = ra
        comp_node[ra] = nxt
        nxt += 1
    root = nxt - 1

    # ---- condensed tree with excess-of-mass stability (lambda = 1/distance) --
    def lam(w):
        return 1.0 / max(float(w), 1e-9)

    clusters = []          # dicts: top(node), death(w), stability, children(list idx)
    # walk each cluster's chain of nodes iteratively
    stack = [(root, np.inf, None)]             # (top node, created_w, parent cluster idx)
    while stack:
        top, created_w, par = stack.pop()
        ci = len(clusters)
        c = {"top": top, "death": created_w, "stability": 0.0, "children": [],
             "drops": [],                      # (side-branch node, fall-out scale)
             "lam_birth": (0.0 if not np.isfinite(created_w) else lam(created_w))}
        clusters.append(c)
        if par is not None:
            clusters[par]["children"].append(ci)
        v = top
        while True:
            c["core"] = v                      # last chain node = the cluster's core
            if left[v] < 0:                    # single leaf: falls out at its own scale
                c["stability"] += lam(weight[v] if weight[v] > 0 else c["lam_birth"]) - c["lam_birth"]
                break
            a, b = left[v], right[v]
            ca_, cb_ = count[a], count[b]
            w = weight[v]
            if ca_ >= min_size and cb_ >= min_size:
                # real split: all current points leave this cluster here
                c["stability"] += count[v] * (lam(w) - c["lam_birth"])
                stack.append((a, w, ci))
                stack.append((b, w, ci))
                break
            if ca_ < min_size and cb_ < min_size:
                # dissolves: remaining points leave around this node's scale
                c["stability"] += count[v] * (lam(w) - c["lam_birth"])
                break
            big, small = (a, b) if ca_ >= cb_ else (b, a)
            c["stability"] += count[small] * (lam(w) - c["lam_birth"])   # side branch falls out
            c["drops"].append((small, w))
            v = big                                                      # cluster continues

    # excess-of-mass selection, bottom-up (children of a cluster index are larger)
    selected = [False] * len(clusters)
    subtree_S = [0.0] * len(clusters)
    for ci in range(len(clusters) - 1, -1, -1):
        c = clusters[ci]
        child_S = sum(subtree_S[k] for k in c["children"])
        if not np.isfinite(c["death"]):
            selected[ci] = False                                   # the root/bulk is not a pocket
            subtree_S[ci] = child_S
        elif c["stability"] >= child_S:
            selected[ci] = True
            subtree_S[ci] = c["stability"]
            def _deselect(k):
                for kk in clusters[k]["children"]:
                    selected[kk] = False
                    _deselect(kk)
            _deselect(ci)
        else:
            subtree_S[ci] = child_S

    def leaves(node):
        out, st = [], [node]
        while st:
            v = st.pop()
            if left[v] < 0:
                out.append(v)
            else:
                st.extend((left[v], right[v]))
        return np.array(out)

    out = []
    for ci, c in enumerate(clusters):
        if not selected[ci] or not np.isfinite(c["death"]):
            continue
        core = c.get("core", c["top"])
        birth = float(first_w[c["top"]])
        death = float(c["death"])
        if not np.isfinite(birth) or death <= birth:
            continue
        # members = the chain's core plus side branches that fell out in the tight
        # regime (below the geometric mean of birth and death) — bulk stragglers
        # that only attached near the merge scale are not counted as pocket members.
        thr = float(np.sqrt(max(birth, 1e-9) * death))
        nodes = [core] + [nd for nd, w in c["drops"] if w <= thr]
        mem = np.unique(np.concatenate([leaves(nd) for nd in nodes]))
        out.append({
            "size": int(len(mem)),
            "birth": birth,
            "death": death,
            "prominence": float(death - birth),
            "members": idx_map[mem],
        })
    out.sort(key=lambda p: -p["prominence"])
    return out[:max_pockets], idx_map
