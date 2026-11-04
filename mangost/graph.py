"""Spatial graph construction (one rule for every dataset, dataset-specific values).

1. radius graph: neighbours within ``radius`` x the section's median nearest-neighbour distance (spot pitch);
2. optional expression pruning: keep an edge when the cosine of the two spots' log-normalised expression is at least
   ``cut``, or drop the ``cut_quantile`` fraction of least similar edges of the section;
3. degree floor: pruned edges are restored, most similar first, until every spot has ``min_degree`` neighbours;
4. optional input smoothing: expression averaged over ``hops`` steps of the graph before the encoder.

The Laplacian positional encoding (16 smallest non-trivial eigenvectors) is computed from the final graph.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def spot_pitch(xy: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors
    xy = np.asarray(xy, np.float64)[:, :2]
    return float(np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]))


def radius_edges(xy: np.ndarray, radius: float = 1.5, pitch: float | None = None) -> np.ndarray:
    """Directed edges (both ways, no self loops) between spots closer than ``radius`` x pitch."""
    from sklearn.neighbors import NearestNeighbors
    xy = np.asarray(xy, np.float64)[:, :2]
    pitch = spot_pitch(xy) if pitch is None else pitch
    G = NearestNeighbors(radius=radius * pitch).fit(xy).radius_neighbors_graph(xy, mode="connectivity")
    G = ((G + G.T) > 0).tolil(); G.setdiag(0); G = G.tocoo()
    ei = np.stack([G.row, G.col]).astype(np.int64)
    return ei[:, ei[0] != ei[1]]


def expression_cut(ei: np.ndarray, X: np.ndarray, cut: float | None = None, cut_quantile: float | None = None,
                   min_degree: int = 2) -> np.ndarray:
    """Expression pruning with a degree floor. ``cut``: cosine threshold; ``cut_quantile``: drop this fraction of the
    least similar edges (the threshold is the section's own cosine quantile).

    The floor follows the code used for the paper: degrees are counted over directed edges (each undirected edge counts
    twice for its endpoints), so ``min_degree=2`` guarantees every spot keeps at least one neighbour."""
    ei = np.asarray(ei, np.int64); Xn = np.asarray(X, np.float64); Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
    sim = (Xn[ei[0]] * Xn[ei[1]]).sum(1)
    if cut_quantile is not None:
        a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1]); pair_sim = sim[a < b]
        cut = float(np.quantile(pair_sim, cut_quantile)) if pair_sim.size else -1.0
    if cut is None:
        return ei
    keep = sim >= cut
    if min_degree > 0:
        n = int(ei.max()) + 1 if ei.size else 0; a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1]); key = a * n + b
        deg = np.bincount(np.concatenate([ei[0][keep], ei[1][keep]]), minlength=n)
        und: dict[int, list[int]] = {}
        for col in range(ei.shape[1]):
            und.setdefault(int(key[col]), []).append(col)
        for col in np.argsort(-sim):
            if keep[col]:
                continue
            u, v = int(ei[0, col]), int(ei[1, col])
            if deg[u] < min_degree or deg[v] < min_degree:
                for c in und[int(key[col])]:
                    keep[c] = True
                deg[u] += 1; deg[v] += 1
    return ei[:, keep]


def laplacian_pe(num_nodes: int, edges: np.ndarray, k: int = 16) -> np.ndarray:
    """The k eigenvectors with the smallest positive eigenvalues of the symmetric normalised Laplacian (null space skipped,
    sign fixed so the largest-magnitude entry is positive). ``edges``: undirected pairs [2, E]."""
    pe = np.zeros((num_nodes, k), np.float32)
    if edges.size == 0:
        return pe
    a, b = edges
    A = sp.coo_matrix((np.ones(a.size * 2), (np.r_[a, b], np.r_[b, a])), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1.0; A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel(); keep = np.flatnonzero(deg > 0)
    if keep.size < 3:
        return pe
    A = A[keep][:, keep]; d = np.asarray(A.sum(1)).ravel(); Dm = sp.diags(1.0 / np.sqrt(d))
    L = (sp.eye(keep.size) - Dm @ A @ Dm).toarray()
    vals, vecs = np.linalg.eigh(L)
    vecs = vecs[:, np.flatnonzero(vals > 1e-6)[:k]]
    for c in range(vecs.shape[1]):
        mag = np.abs(vecs[:, c]); j = int(np.flatnonzero(mag >= mag.max() - 1e-6)[0])
        if vecs[j, c] < 0:
            vecs[:, c] = -vecs[:, c]
    pe[keep, :vecs.shape[1]] = vecs.astype(np.float32)
    return pe


def smooth_hops(X: np.ndarray, edge_index: np.ndarray, hops: int) -> np.ndarray:
    """X <- (D^-1 (A + I))^hops X."""
    X = np.asarray(X, np.float64)
    if hops <= 0:
        return X
    ei = np.asarray(edge_index); n = X.shape[0]
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(np.float64) + sp.eye(n)
    P = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    for _ in range(hops):
        X = P @ X
    return np.asarray(X)


def build_graph(xy, X, radius: float = 1.5, cut: float | None = None, cut_quantile: float | None = None, min_degree: int = 2):
    """Edges (both directions) and the 16-dim Laplacian PE of one section."""
    ei = radius_edges(xy, radius)
    if cut is not None or cut_quantile is not None:
        ei = expression_cut(ei, X, cut=cut, cut_quantile=cut_quantile, min_degree=min_degree)
    und = np.unique(np.sort(ei.T, 1), axis=0).T if ei.size else ei
    return ei, laplacian_pe(np.asarray(xy).shape[0], und, 16)
