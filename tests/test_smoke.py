"""Fast end-to-end check on a tiny synthetic dataset (CPU, a few epochs)."""
import numpy as np

import mangost as mg


def test_fit_small():
    A = mg.datasets.make_synthetic(n_sections=2, n_side=14, n_genes=80, markers_per_region=5, module_size=6, seed=1)
    m = mg.MANGOST(K=4, epochs=3, hidden=64, width=32, d_gene=16, device="cpu", verbose=False).fit(A, n_hvg=None)
    assert A.obsm["X_mangost"].shape == (A.n_obs, 30)
    alpha = A.obsm["mangost_alpha"]
    assert np.allclose(alpha.sum(1), 1.0, atol=1e-4)
    W = m.network(0)
    assert np.allclose(np.diag(W.values), 0.0)
    mg.tl.cluster_domains(A, groupby="region", method="gmm")
    s = mg.tl.score(A, "mangost_domain", "region")
    assert {"ARI", "NMI"} <= set(s.columns)


def test_graph_rules():
    xy = np.stack(np.meshgrid(np.arange(10), np.arange(10)), -1).reshape(-1, 2).astype(float)
    ei = mg.graph.radius_edges(xy, radius=1.5)
    assert ei.shape[0] == 2 and (ei[0] != ei[1]).all()
    X = np.random.default_rng(0).random((100, 20))
    pruned = mg.graph.expression_cut(ei, X, cut_quantile=0.4)
    deg = np.bincount(pruned[0], minlength=100)
    assert pruned.shape[1] < ei.shape[1] and deg.min() >= 1   # paper rule: at least one neighbour kept
