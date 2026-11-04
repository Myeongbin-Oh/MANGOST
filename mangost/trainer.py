"""High-level interface: ``MANGOST(...).fit(adata)``.

The training loop follows the reference implementation used for the paper:
* all sections of a dataset are trained jointly, one full-batch optimiser step per section, sections in random order
  every epoch;
* 20% of the entries of each spot's expression are masked; the same entries are masked in the neighbour-smoothed,
  standardised network target;
* AdamW (lr 1e-3, weight decay 1e-2), linear warm-up for 5 epochs then cosine decay, gradient norm clipped at 1.0.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .graph import build_graph, smooth_hops
from .losses import expert_errors, gate_cross_entropy, masked_recon, min_usage_hinge, responsibility_loss, usage_smoothness
from .model import MANGOSTNet, allowed_masks


# ----------------------------------------------------------------------------------------------- preprocessing
def lognormalise(counts) -> np.ndarray:
    import scipy.sparse as sp
    C = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    C = C.astype(np.float64); lib = C.sum(1, keepdims=True); lib[lib == 0] = 1.0
    return np.log1p(C / lib * 1e4).astype(np.float32)


def select_hvg(adata, n_hvg: int, section_key: str, counts_layer: str | None) -> list[str]:
    """seurat_v3 highly variable genes with the section as batch key, genes variable in more sections first."""
    import anndata as ad
    import scanpy as sc
    X = adata.layers[counts_layer] if counts_layer else adata.X
    a = ad.AnnData(X.copy()); a.var_names = adata.var_names; a.obs["section"] = adata.obs[section_key].astype(str).values
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=min(n_hvg, a.n_vars), batch_key="section")
    v = a.var.loc[a.var["highly_variable"]].sort_values(["highly_variable_nbatches", "highly_variable_rank"], ascending=[False, True])
    return [str(g) for g in v.index]


def neighbour_smoothed(X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """(I + A) X / (degree + 1), section mean removed."""
    N = X.shape[0]; src, dst = edge_index[0], edge_index[1]
    out = X.clone().index_add_(0, dst, X[src])
    deg = torch.zeros(N, device=X.device, dtype=X.dtype).index_add_(0, dst, torch.ones(dst.shape[0], device=X.device, dtype=X.dtype))
    out = out / (deg + 1).unsqueeze(1)
    return out - out.mean(0, keepdim=True)


def mask_rows(X: torch.Tensor, frac: float, gen: torch.Generator):
    m = (torch.rand(X.shape, generator=gen, device="cpu") < frac).to(X.device)
    return X.masked_fill(m, 0.0), m


def lr_factor(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(total - warmup, 1)))


def set_deterministic(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass


@dataclass
class _Section:
    name: str
    index: np.ndarray          # row positions in the AnnData
    X: torch.Tensor
    ei: torch.Tensor
    pe: torch.Tensor
    allowed_self: torch.Tensor
    allowed_nb: torch.Tensor
    sec_idx: torch.Tensor
    xbar: torch.Tensor | None = None


# ----------------------------------------------------------------------------------------------- model
class MANGOST:
    """🥭 MANGOST: spatial domains and region-specific gene networks from multi-section spatial transcriptomics.

    Parameters
    ----------
    K : number of shared gene networks
    epochs, lr, weight_decay, warmup : optimisation
    hidden, embed, width, d_gene : encoder width, spatial-embedding size, transformer width, gene-embedding rank
    mask_frac : fraction of masked expression entries per spot
    tau : temperature of the likelihood mixture
    w_gate, w_floor : weights of the gate alignment and minimum-usage terms
    net_detect : network genes = genes detected in at least this fraction of spots (averaged over sections)
    seed, deterministic, device : reproducibility and hardware
    """

    def __init__(self, K: int = 10, epochs: int = 600, lr: float = 1e-3, weight_decay: float = 1e-2, warmup: int = 5,
                 hidden: int = 512, embed: int = 30, width: int = 128, d_gene: int = 64, mask_frac: float = 0.2,
                 tau: float = 0.1, w_gate: float = 0.1, w_floor: float = 1.0, net_detect: float = 0.05,
                 seed: int = 0, deterministic: bool = True, device: str | None = None, verbose: bool = True):
        self.cfg = dict(K=K, epochs=epochs, lr=lr, weight_decay=weight_decay, warmup=warmup, hidden=hidden, embed=embed,
                        width=width, d_gene=d_gene, mask_frac=mask_frac, tau=tau, w_gate=w_gate, w_floor=w_floor,
                        net_detect=net_detect, seed=seed, deterministic=deterministic)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.net: MANGOSTNet | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ data
    def _prepare(self, adata, section_key, spatial_key, counts_layer, n_hvg, radius, cut, cut_quantile, hops):
        sections = list(dict.fromkeys(adata.obs[section_key].astype(str).values))
        genes = select_hvg(adata, n_hvg, section_key, counts_layer) if (n_hvg and adata.n_vars > n_hvg) else list(map(str, adata.var_names))
        gidx = adata.var_names.get_indexer(genes)
        counts = (adata.layers[counts_layer] if counts_layer else adata.X)
        prepared, Xs = [], []
        for si, name in enumerate(sections):
            rows = np.flatnonzero(adata.obs[section_key].astype(str).values == name)
            X = lognormalise(counts[rows][:, gidx]); xy = np.asarray(adata.obsm[spatial_key])[rows, :2]
            ei, pe = build_graph(xy, X, radius=radius, cut=cut, cut_quantile=cut_quantile)
            if hops > 0:
                X = smooth_hops(X, ei, hops).astype(np.float32)
            Xs.append(X)
            Xt = torch.as_tensor(X, device=self.device); eit = torch.as_tensor(ei, device=self.device)
            a_self, a_nb = allowed_masks(eit, len(rows), self.device)
            prepared.append(_Section(name, rows, Xt, eit, torch.as_tensor(pe, device=self.device), a_self, a_nb,
                                     torch.full((len(rows),), si, device=self.device, dtype=torch.long)))
            if self.verbose:
                print(f"  {name}: {len(rows)} spots, {ei.shape[1] // 2} edges (mean degree {ei.shape[1] / len(rows):.2f})")
        det = np.mean([(X > 0).mean(0) for X in Xs], 0)
        net_idx = np.flatnonzero(det >= self.cfg["net_detect"])
        ni = torch.as_tensor(net_idx, device=self.device, dtype=torch.long)
        for P in prepared:
            xb = neighbour_smoothed(P.X, P.ei)[:, ni]
            P.xbar = xb / xb.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return prepared, genes, net_idx

    # ------------------------------------------------------------------ training
    def _step(self, P: _Section, gen):
        c = self.cfg
        xm, mask = mask_rows(P.X, c["mask_frac"], gen); mask_n = mask[:, self.net_idx_t]
        xbar_obs = P.xbar.masked_fill(mask_n, 0.0)
        out = self.net(xm, P.X, xbar_obs, P.ei, P.pe, P.allowed_self, P.allowed_nb, P.sec_idx)
        E = expert_errors(self.net.nets, P.xbar, mask_n)
        net_loss, resp = responsibility_loss(E, out["alpha"], c["tau"])
        parts = {"spot": masked_recon(out["x_hat"], P.X, mask),
                 "net": net_loss,
                 "gate": c["w_gate"] * gate_cross_entropy(resp, out["alpha"]),
                 "smooth": usage_smoothness(out["alpha_soft"], P.ei),
                 "use": c["w_floor"] * min_usage_hinge(out["alpha_soft"])}
        return parts

    def fit(self, adata, section_key: str = "section", spatial_key: str = "spatial", counts_layer: str | None = None,
            n_hvg: int | None = 4000, radius: float = 1.5, cut: float | None = None, cut_quantile: float | None = None,
            hops: int = 0):
        """Train on all sections of ``adata`` (raw counts in ``adata.X`` or ``adata.layers[counts_layer]``).

        Graph options (Supplementary Table S2 of the paper): ``radius`` x spot pitch; expression pruning by ``cut``
        (cosine) or ``cut_quantile`` (fraction of least similar edges removed); ``hops`` of input smoothing.
        Results are written to ``adata.obsm['X_mangost']`` (spatial embedding s') and ``adata.obsm['mangost_alpha']``
        (routing weights).
        """
        c = self.cfg
        torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
        if c["deterministic"]:
            set_deterministic(c["seed"])
        if self.verbose:
            print("🥭 MANGOST: preparing sections")
        prepared, genes, net_idx = self._prepare(adata, section_key, spatial_key, counts_layer, n_hvg, radius, cut, cut_quantile, hops)
        self.genes, self.net_genes = genes, [genes[i] for i in net_idx]
        self.net_idx_t = torch.as_tensor(net_idx, device=self.device, dtype=torch.long)
        self.net = MANGOSTNet(len(genes), len(prepared), len(net_idx), K=c["K"], d_gene=c["d_gene"], hidden=c["hidden"],
                              embed=c["embed"], width=c["width"]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        gen = torch.Generator().manual_seed(c["seed"])
        if self.verbose:
            print(f"🥭 training: {len(prepared)} sections, {len(genes)} genes, {len(net_idx)} network genes, K={c['K']}, device {self.device}")
        self.net.train()
        for epoch in range(c["epochs"]):
            for g in opt.param_groups:
                g["lr"] = c["lr"] * lr_factor(epoch, c["warmup"], c["epochs"])
            sums: dict[str, float] = {}
            for i in np.random.permutation(len(prepared)):
                parts = self._step(prepared[i], gen)
                loss = sum(parts.values())
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
            row = {"epoch": epoch, **{k: v / len(prepared) for k, v in sums.items()}}
            self.history.append(row)
            if self.verbose and (epoch % 50 == 0 or epoch == c["epochs"] - 1):
                print(f"  epoch {epoch:4d}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"))
        self._encode(adata, prepared)
        return self

    @torch.no_grad()
    def _encode(self, adata, prepared):
        self.net.eval()
        S = np.zeros((adata.n_obs, self.cfg["embed"]), np.float32); A = np.zeros((adata.n_obs, self.cfg["K"]), np.float32)
        for P in prepared:
            out = self.net(P.X, P.X, P.xbar, P.ei, P.pe, P.allowed_self, P.allowed_nb, P.sec_idx)
            S[P.index] = out["s"].cpu().numpy(); A[P.index] = out["alpha"].cpu().numpy()
        adata.obsm["X_mangost"] = S; adata.obsm["mangost_alpha"] = A
        used = self.used_networks(A)
        adata.uns["mangost"] = {"network_genes": self.net_genes, "used_networks": used, "config": dict(self.cfg)}
        if self.verbose:
            print(f"🥭 done: networks in use {used}")
        self.net.train()

    # ------------------------------------------------------------------ networks
    @staticmethod
    def used_networks(alpha: np.ndarray, threshold: float = 0.05) -> list[int]:
        """Networks with mean routing weight above ``threshold``."""
        return [int(k) for k in np.flatnonzero(np.asarray(alpha).mean(0) > threshold)]

    def network(self, k: int) -> pd.DataFrame:
        """W_k as a genes x genes DataFrame (zero diagonal)."""
        W = self.net.nets.matrix(k).cpu().numpy()
        return pd.DataFrame(W, index=self.net_genes, columns=self.net_genes)

    def network_deviation(self, k: int, networks: list[int] | None = None) -> pd.DataFrame:
        """Delta W_k = W_k - mean of the networks in use: what is specific to network k."""
        networks = networks if networks is not None else list(range(self.cfg["K"]))
        Wm = np.mean([self.net.nets.matrix(j).cpu().numpy() for j in networks], 0)
        return self.network(k) - Wm

    def hub_genes(self, k: int, n: int = 15, networks: list[int] | None = None) -> pd.Series:
        """Genes with the largest total region-specific association, sum_g' |Delta W_k[g, g']|."""
        D = self.network_deviation(k, networks)
        return D.abs().sum(1).sort_values(ascending=False).head(n)

    def top_edges(self, k: int, n: int = 30, networks: list[int] | None = None) -> pd.DataFrame:
        """The ``n`` strongest positive region-specific associations of network k."""
        D = self.network_deviation(k, networks).values; g = self.net_genes
        iu = np.triu_indices(len(g), 1); w = D[iu]; o = np.argsort(-w)[:n]
        return pd.DataFrame({"gene_a": [g[i] for i in iu[0][o]], "gene_b": [g[j] for j in iu[1][o]], "weight": w[o]})

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        torch.save({"state": self.net.state_dict(), "cfg": self.cfg, "genes": self.genes, "net_genes": self.net_genes,
                    "n_sections": int(self.net.enc.section_emb.num_embeddings)}, path)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ck["cfg"], device=device, verbose=False)
        c = m.cfg; m.genes, m.net_genes = ck["genes"], ck["net_genes"]
        m.net = MANGOSTNet(len(m.genes), ck["n_sections"], len(m.net_genes), K=c["K"], d_gene=c["d_gene"], hidden=c["hidden"],
                           embed=c["embed"], width=c["width"]).to(m.device)
        m.net.load_state_dict(ck["state"]); return m
