"""Downstream analysis: spatial domains, region usage of the networks, scoring."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def mclust(x: np.ndarray, n_clusters: int, seed: int = 2020, model: str = "EEE") -> np.ndarray:
    """R ``mclust::Mclust`` (the clustering used in the paper). Requires R with the mclust and jsonlite packages."""
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript not found; install R + mclust, or use method='gmm'")
    with tempfile.TemporaryDirectory(prefix="mangost_mclust_") as d:
        d = Path(d); np.savetxt(d / "x.tsv", np.asarray(x, np.float64), delimiter="\t", fmt="%.10g")
        code = f"""
        suppressPackageStartupMessages(library(mclust))
        x <- as.matrix(read.table("{(d / 'x.tsv').as_posix()}", sep="\\t", header=FALSE))
        set.seed({int(seed)})
        res <- NULL
        for (m in c("{model}", "EEI", "VVI", "EII", "VII")) {{
          res <- tryCatch(Mclust(x, G={int(n_clusters)}, modelNames=m, verbose=FALSE), error=function(e) NULL)
          if (!is.null(res)) break
        }}
        write.table(as.integer(res$classification), file="{(d / 'lab.txt').as_posix()}", row.names=FALSE, col.names=FALSE)
        """
        subprocess.run(["Rscript", "-e", code], check=True, capture_output=True, text=True)
        return np.atleast_1d(np.loadtxt(d / "lab.txt", dtype=np.int64)) - 1


def gmm(x: np.ndarray, n_clusters: int, seed: int = 2020) -> np.ndarray:
    """Python fallback: Gaussian mixture with one shared covariance (the EEE model of mclust)."""
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_clusters, covariance_type="tied", random_state=seed, n_init=3).fit_predict(np.asarray(x, np.float64))


def cluster_domains(adata, n_clusters: int | dict | None = None, groupby: str | None = None, section_key: str = "section",
                    method: str = "mclust", key: str = "mangost_domain", seed: int = 2020):
    """Spatial domains: per-section clustering of the spatial embedding ``adata.obsm['X_mangost']``.

    ``n_clusters``: an int, a {section: k} dict, or None to take each section's number of ``groupby`` classes
    (the evaluation protocol of the paper)."""
    labels = np.empty(adata.n_obs, dtype=object); sec = adata.obs[section_key].astype(str).values
    fn = mclust if method == "mclust" else gmm
    for s in dict.fromkeys(sec):
        rows = np.flatnonzero(sec == s)
        if isinstance(n_clusters, dict): k = n_clusters[s]
        elif n_clusters is not None: k = int(n_clusters)
        else: k = adata.obs[groupby].iloc[rows].nunique()
        labels[rows] = [f"{s}_{c}" for c in fn(adata.obsm["X_mangost"][rows], k, seed=seed)]
    adata.obs[key] = pd.Categorical(labels)
    return adata


def match_to_reference(adata, key: str, reference: str, section_key: str = "section", out_key: str | None = None):
    """Rename each section's clusters to the reference class they overlap most (one-to-one Hungarian), for colours."""
    from scipy.optimize import linear_sum_assignment
    out = np.empty(adata.n_obs, dtype=object); sec = adata.obs[section_key].astype(str).values
    for s in dict.fromkeys(sec):
        rows = np.flatnonzero(sec == s); a = adata.obs[key].astype(str).values[rows]; b = adata.obs[reference].astype(str).values[rows]
        A, B = sorted(set(a)), sorted(set(b)); M = np.array([[np.sum((a == i) & (b == j)) for j in B] for i in A])
        r, c = linear_sum_assignment(-M); name = {A[i]: B[j] for i, j in zip(r, c)}
        out[rows] = [name.get(x, "unmatched") for x in a]
    adata.obs[out_key or f"{key}_matched"] = pd.Categorical(out)
    return adata


def score(adata, key: str, reference: str, section_key: str = "section") -> pd.DataFrame:
    """Per-section ARI and NMI of ``key`` against ``reference`` (labelled spots only)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows = []
    sec = adata.obs[section_key].astype(str).values
    for s in dict.fromkeys(sec):
        m = (sec == s) & adata.obs[reference].notna().values
        a, b = adata.obs[reference].astype(str).values[m], adata.obs[key].astype(str).values[m]
        rows.append({"section": s, "ARI": adjusted_rand_score(a, b), "NMI": normalized_mutual_info_score(a, b)})
    df = pd.DataFrame(rows).set_index("section"); df.loc["mean"] = df.mean()
    return df.round(3)


def region_usage(adata, groupby: str, networks: list[int] | None = None) -> pd.DataFrame:
    """Mean routing weight of every network in every region (raw means; rows need not sum to one)."""
    A = adata.obsm["mangost_alpha"]; networks = networks if networks is not None else adata.uns["mangost"]["used_networks"]
    df = pd.DataFrame(A[:, networks], columns=[f"net {k}" for k in networks], index=adata.obs_names)
    return df.groupby(adata.obs[groupby].astype(str).values).mean()


def regional_sets(usage: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Network -> regions with mean routing weight >= threshold (falls back to the highest-usage region)."""
    out = {}
    for col in usage.columns:
        s = usage[col]; r = list(s.index[s >= threshold])
        out[col] = r if r else [s.idxmax() + " (fallback)"]
    return out


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))
