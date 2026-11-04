"""Plots: spatial maps of domains and routing weights, region usage heat map, network sub-graphs."""
from __future__ import annotations

import numpy as np
import pandas as pd

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
           "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]


def _axes(n, size=3.0):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, n, figsize=(size * n, size), squeeze=False)
    return fig, axs[0]


def _clean(ax, title):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(title, fontsize=11)


def spatial(adata, color: str, section_key: str = "section", sections=None, spot_size: float = 8, size: float = 3.0):
    """Categorical map of ``adata.obs[color]`` per section, one colour per category shared across sections."""
    sec = adata.obs[section_key].astype(str).values; sections = sections or list(dict.fromkeys(sec))
    cats = sorted(adata.obs[color].astype(str).unique()); cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}
    fig, axs = _axes(len(sections), size)
    for ax, s in zip(axs, sections):
        m = sec == s; xy = adata.obsm["spatial"][m]
        ax.scatter(xy[:, 0], xy[:, 1], c=[cmap[c] for c in adata.obs[color].astype(str).values[m]], s=spot_size, linewidths=0)
        _clean(ax, s)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", markerfacecolor=cmap[c], markeredgecolor="none", label=c) for c in cats],
               loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(color, y=1.02); fig.tight_layout()
    return fig


def routing(adata, section: str, networks=None, section_key: str = "section", spot_size: float = 8, size: float = 2.6):
    """Routing weight alpha of each network on one section (viridis, shared 0-1 scale)."""
    import matplotlib.pyplot as plt
    networks = networks if networks is not None else adata.uns["mangost"]["used_networks"]
    m = adata.obs[section_key].astype(str).values == section; xy = adata.obsm["spatial"][m]; A = adata.obsm["mangost_alpha"][m]
    fig, axs = _axes(len(networks), size)
    for ax, k in zip(axs, networks):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=A[:, k], cmap="viridis", vmin=0, vmax=1, s=spot_size, linewidths=0)
        _clean(ax, f"network {k}")
    fig.colorbar(sc, ax=list(axs), fraction=0.02, pad=0.01, label="routing weight α")
    fig.suptitle(section, y=1.02)
    return fig


def usage_heatmap(usage: pd.DataFrame, size=(4.5, 3.2)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(usage.values, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(usage.shape[1])); ax.set_xticklabels(usage.columns, rotation=45, ha="right")
    ax.set_yticks(range(usage.shape[0])); ax.set_yticklabels(usage.index)
    for i in range(usage.shape[0]):
        for j in range(usage.shape[1]):
            v = usage.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "#333333")
    fig.colorbar(im, ax=ax, label="mean routing weight α"); ax.set_title("Network usage by region")
    fig.tight_layout(); return fig


def network_graph(edges: pd.DataFrame, highlight=None, title: str = "", size: float = 4.0):
    """Circular drawing of a network's strongest region-specific edges (``MANGOST.top_edges``); ``highlight``: genes in pink."""
    import matplotlib.pyplot as plt
    genes = list(dict.fromkeys(list(edges.gene_a) + list(edges.gene_b))); n = len(genes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False); pos = {g: (np.cos(a), np.sin(a)) for g, a in zip(genes, ang)}
    fig, ax = plt.subplots(figsize=(size, size)); wmax = edges.weight.max()
    for _, e in edges.iterrows():
        (x1, y1), (x2, y2) = pos[e.gene_a], pos[e.gene_b]
        ax.plot([x1, x2], [y1, y2], color="#2F7F9D", lw=0.5 + 3 * e.weight / wmax, alpha=0.8, zorder=1)
    hl = set() if highlight is None else set(map(str, highlight))
    for g in genes:
        x, y = pos[g]; on = g in hl
        ax.scatter([x], [y], s=110, color="#EFB3C2" if on else "#DDE4EC", edgecolor="#A84A63" if on else "#6E7B8B", zorder=2)
        ax.text(1.15 * x, 1.15 * y, g, ha="center", va="center", fontsize=7, color="#A84A63" if on else "#2B2B2B", fontweight="bold" if on else "normal")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.axis("off"); ax.set_title(title)
    return fig
