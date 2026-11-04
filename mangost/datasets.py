"""Synthetic multi-section spatial transcriptomics data for the tutorial.

The generator mimics a layered tissue (for example a cortical column) cut into several sections:

* every section is a hexagonal spot array with curved, laminar regions;
* each region has its own marker genes (higher mean expression), and
* each region has its own **gene co-expression programme**: a module of genes that vary together
  only inside that region. These region-specific gene-gene relationships are what MANGOST's
  routed gene networks are designed to recover;
* a global module, library-size differences and a mild per-section batch effect are added.

Counts are drawn from a negative binomial, so the usual normalise + log1p preprocessing applies.
"""
from __future__ import annotations

import numpy as np

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None


def _hex_grid(n_side: int, rng: np.random.Generator, angle: float) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    xy = np.stack([xs.ravel() + 0.5 * (ys.ravel() % 2), ys.ravel() * np.sqrt(3) / 2], 1).astype(float)
    xy -= xy.mean(0)
    keep = np.linalg.norm(xy / (n_side / 2), axis=1) < 1.0          # round tissue outline
    xy = xy[keep]
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return xy @ rot.T + rng.normal(0, 0.03, xy.shape)


def make_synthetic(
    n_sections: int = 3,
    n_side: int = 40,
    n_regions: int = 5,
    n_genes: int = 300,
    markers_per_region: int = 12,
    module_size: int = 15,
    seed: int = 0,
):
    """Return an AnnData with raw counts in ``X`` and ``obs['section']``, ``obs['region']``, ``obsm['spatial']``.

    Parameters
    ----------
    n_sections : number of tissue sections (all share the same regions and gene programmes)
    n_side : side length of the hexagonal spot array (about 0.8 * n_side**2 spots per section)
    n_regions : number of laminar regions
    n_genes : number of genes
    markers_per_region : genes with higher mean expression in each region
    module_size : genes in each region-specific co-expression module
    seed : random seed
    """
    if ad is None:
        raise ImportError("anndata is required: pip install anndata")
    rng = np.random.default_rng(seed)
    names = [f"Region_{chr(65 + r)}" for r in range(n_regions)]
    genes = [f"Gene{g:03d}" for g in range(n_genes)]

    # gene roles: markers, region modules, one global module, the rest background
    perm = rng.permutation(n_genes)
    markers = [perm[r * markers_per_region:(r + 1) * markers_per_region] for r in range(n_regions)]
    off = n_regions * markers_per_region
    modules = [perm[off + r * module_size: off + (r + 1) * module_size] for r in range(n_regions)]
    off2 = off + n_regions * module_size
    global_module = perm[off2: off2 + module_size]
    base = rng.gamma(2.0, 0.6, n_genes) + 0.2                         # baseline mean per gene

    X_all, sec_all, reg_all, xy_all = [], [], [], []
    for s in range(n_sections):
        xy = _hex_grid(n_side, rng, angle=rng.uniform(-0.4, 0.4))
        n = len(xy)
        # curved laminar boundaries along y, slightly different per section
        bend = 0.08 * n_side * np.sin(xy[:, 0] / n_side * np.pi + rng.uniform(0, 1))
        depth = (xy[:, 1] + bend - xy[:, 1].min()) / (np.ptp(xy[:, 1]) + 1e-9)
        cuts = np.linspace(0, 1, n_regions + 1)[1:-1] + rng.uniform(-0.04, 0.04, n_regions - 1)   # even layers, jittered per section
        region = np.digitize(depth, cuts)

        log_mu = np.tile(np.log(base), (n, 1))
        for r in range(n_regions):
            in_r = region == r
            log_mu[np.ix_(in_r, markers[r])] += 1.2                    # marker shift
            # region-specific programme: one latent factor shared by the module genes, only inside region r
            f = rng.normal(0, 1, in_r.sum())[:, None]
            load = rng.uniform(0.6, 1.0, module_size)[None, :]
            log_mu[np.ix_(in_r, modules[r])] += 0.9 * f * load
        g = rng.normal(0, 1, n)[:, None]                                # programme active everywhere
        log_mu[:, global_module] += 0.5 * g
        log_mu += rng.normal(0, 0.15, n_genes)[None, :]                 # per-section batch effect
        lib = rng.lognormal(0, 0.25, n)[:, None]                        # library size
        mu = np.exp(log_mu) * lib
        theta = 5.0                                                     # negative-binomial dispersion
        counts = rng.negative_binomial(theta, theta / (theta + mu))

        X_all.append(counts.astype(np.float32)); xy_all.append(xy)
        sec_all += [f"section{s + 1}"] * n; reg_all += [names[k] for k in region]

    A = ad.AnnData(np.vstack(X_all))
    A.var_names = genes
    A.obs_names = [f"spot{i}" for i in range(A.n_obs)]
    A.obs["section"] = np.array(sec_all)
    A.obs["region"] = np.array(reg_all)
    A.obsm["spatial"] = np.vstack(xy_all).astype(np.float32)
    A.uns["synthetic_truth"] = {
        "markers": {names[r]: [genes[i] for i in markers[r]] for r in range(n_regions)},
        "modules": {names[r]: [genes[i] for i in modules[r]] for r in range(n_regions)},
        "global_module": [genes[i] for i in global_module],
    }
    return A
