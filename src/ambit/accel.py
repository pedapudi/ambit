"""Optional CUDA / torch acceleration backend.

ambit's hot kernels are GEMM-heavy (the streaming covariance X^T X, PCA SVD, the
random-pair cosine, brute-force kNN). On CPU those already dispatch to BLAS, so
this module exists for the case where a GPU is worth it at 3-5M+ scale: it mirrors
each kernel on a torch device (cuda / mps / cpu) and returns numpy at the
boundaries, so the rest of ambit (figures, metrics, render) is unchanged.

Activated by `device != "cpu"` on `scan()` / `build_ctx()`:
    cpu    -> numpy (no torch needed; the default)
    auto   -> torch, cuda if available else mps else cpu
    cuda   -> torch on CUDA (errors if unavailable)
    mps    -> torch on Apple Metal
    torch  -> torch on CPU (for testing the tensor path without a GPU)

Install with `pip install 'ambit[gpu]'`. For kNN over a very large reservoir or the
full corpus, set `AMBIT_FAISS=1` to use FAISS-GPU (`ambit[faiss]`) instead of the
torch brute-force path.
"""

from __future__ import annotations

import os

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def resolve_device(spec: str) -> str:
    """Map a device spec to a concrete torch device string."""
    import torch
    if spec in ("torch", "cpu-torch"):
        return "cpu"
    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    return spec


class TorchCov:
    """Streaming covariance/sum/norm accumulator on a torch device. The per-chunk
    gram is float32 (tensor-core friendly); totals accumulate in float64 on device."""

    def __init__(self, device: str):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.n = 0
        self.dim = None
        self.sumv = self.XtX = None
        self.ns = self.nsq = 0.0

    def update(self, Xc):
        torch = self.torch
        X = torch.from_numpy(np.array(Xc, dtype=np.float32)).to(self.device, non_blocking=True)
        m, d = X.shape
        if self.dim is None:
            self.dim = d
            self.sumv = torch.zeros(d, dtype=torch.float64, device=self.device)
            self.XtX = torch.zeros((d, d), dtype=torch.float64, device=self.device)
        elif d != self.dim:
            raise ValueError(f"inconsistent embedding dim: {d} != {self.dim}")
        self.sumv += X.sum(0, dtype=torch.float64)
        self.XtX += (X.t() @ X).to(torch.float64)
        nr = torch.linalg.vector_norm(X, dim=1)
        self.ns += float(nr.sum().item())
        self.nsq += float((nr * nr).sum().item())
        self.n += m

    def finalize(self):
        torch = self.torch
        mean = self.sumv / self.n
        cov = (self.XtX - self.n * torch.outer(mean, mean)) / max(1, self.n - 1)
        nm = self.ns / self.n
        nsd = float(np.sqrt(max(0.0, self.nsq / self.n - nm * nm)))
        return self.dim, mean.cpu().numpy(), cov.cpu().numpy(), nm, nsd


def torch_pca(X, k: int, device: str) -> np.ndarray:
    import torch
    dev = torch.device(device)
    T = torch.from_numpy(np.array(X, dtype=np.float32)).to(dev)
    T = T - T.mean(0, keepdim=True)
    # low-rank SVD: just the top-k singular directions (fast on device)
    try:
        U, S, V = torch.svd_lowrank(T, q=min(k + 6, min(T.shape)))
        comps = V[:, :k]
    except Exception:
        _, _, Vh = torch.linalg.svd(T, full_matrices=False)
        comps = Vh[:k].t()
    return (T @ comps).cpu().numpy()


def torch_random_pair_cosine(Xn, n_pairs: int, device: str, seed: int = 0) -> np.ndarray:
    """Xn assumed L2-normalized. Samples random off-diagonal pairs on device."""
    import torch
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    T = torch.from_numpy(np.array(Xn, dtype=np.float32)).to(dev)
    n = T.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=g, device=dev)
    j = torch.randint(0, n, (n_pairs,), generator=g, device=dev)
    keep = i != j
    cos = (T[i[keep]] * T[j[keep]]).sum(1)
    return cos.cpu().numpy().astype(np.float64)


def torch_knn(Xn, k: int, device: str, block: int = 4096):
    """Exact cosine kNN on device, blocked so the (block x n) similarity tile stays
    bounded. Xn assumed L2-normalized. For n in the millions prefer FAISS-GPU."""
    if os.environ.get("AMBIT_FAISS") == "1":
        try:
            return _faiss_knn(Xn, k)
        except Exception:
            pass
    import torch
    dev = torch.device(device)
    T = torch.from_numpy(np.array(Xn, dtype=np.float32)).to(dev)
    n = T.shape[0]
    kk = min(k, n - 1)
    idx = np.empty((n, kk), dtype=np.int64)
    dist = np.empty((n, kk), dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        sims = T[s:e] @ T.t()                                  # (b, n) cosine
        rows = torch.arange(e - s, device=dev)
        sims[rows, torch.arange(s, e, device=dev)] = float("-inf")   # drop self
        vals, ind = torch.topk(sims, kk, dim=1)
        idx[s:e] = ind.cpu().numpy()
        dist[s:e] = (1.0 - vals).cpu().numpy()
    return idx, dist


def _faiss_knn(Xn, k: int):
    import faiss  # ambit[faiss]; faiss-gpu for the GPU index
    X = np.array(Xn, dtype=np.float32)
    n, d = X.shape
    index = faiss.IndexFlatIP(d)                               # inner product = cosine on unit vectors
    if hasattr(faiss, "StandardGpuResources"):
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.add(X)
    sims, ind = index.search(X, min(k, n - 1) + 1)
    return ind[:, 1:], (1.0 - sims[:, 1:]).astype(np.float32)  # drop self
