"""Command-line training: python scripts/run_mangost.py data/mangost_synthetic.h5ad --out results/

Writes <out>/mangost.h5ad (embedding, routing weights, domains), <out>/model.pt and, when --label is given, <out>/scores.csv.
"""
import argparse
from pathlib import Path

import anndata as ad

import mangost as mg


def main():
    p = argparse.ArgumentParser(description="🥭 MANGOST")
    p.add_argument("h5ad", help="AnnData with raw counts, obs[section], obsm[spatial]")
    p.add_argument("--out", default="results")
    p.add_argument("--section-key", default="section")
    p.add_argument("--label", default=None, help="annotation column (only to choose k per section and to score)")
    p.add_argument("--n-clusters", type=int, default=None)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-hvg", type=int, default=4000, help="0 = use all genes (gene panels)")
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--cut", type=float, default=None)
    p.add_argument("--cut-quantile", type=float, default=None)
    p.add_argument("--hops", type=int, default=0)
    p.add_argument("--cluster", choices=["mclust", "gmm"], default="mclust")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(a.h5ad)
    model = mg.MANGOST(K=a.K, epochs=a.epochs, seed=a.seed).fit(
        adata, section_key=a.section_key, n_hvg=a.n_hvg or None, radius=a.radius, cut=a.cut, cut_quantile=a.cut_quantile, hops=a.hops)
    mg.tl.cluster_domains(adata, n_clusters=a.n_clusters, groupby=a.label, section_key=a.section_key, method=a.cluster)
    if a.label:
        mg.tl.score(adata, "mangost_domain", a.label, a.section_key).to_csv(out / "scores.csv")
    model.save(str(out / "model.pt")); adata.write_h5ad(out / "mangost.h5ad")
    print(f"🥭 results in {out}/")


if __name__ == "__main__":
    main()
