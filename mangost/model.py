"""MANGOST network.

Masked graph attention encoder -> spatial transformer -> spatial embedding s'
  * graph decoder reconstructs the expression from s' (spot path)
  * spatially informed router maps s' to routing weights alpha (N x K)
  * K shared low-rank gene networks W_k = offdiag(G_k G_k^T / d), mixed per spot by alpha (network path)

The modules are taken from the research code that produced the paper results, keeping only the
branches of the final configuration. Parameters are created in the same order as in that code
(encoder, spatial block, router, network bank), so a given seed gives the same initialisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as seg_softmax


class GATLayer(nn.Module):
    """One attention aggregation (TiedGAT score): sigmoid(leaky_relu(a_src.h_j + a_dst.h_i)), softmax over the incoming
    edges of i, self loop included. ``h_self`` feeds the self loop and the destination score (the spot's masked row);
    ``h_nb`` feeds the messages from neighbours (their full rows)."""

    def __init__(self, width: int):
        super().__init__()
        self.att_src = nn.Parameter(torch.empty(1, width)); self.att_dst = nn.Parameter(torch.empty(1, width))
        nn.init.xavier_uniform_(self.att_src); nn.init.xavier_uniform_(self.att_dst)

    def forward(self, h_self, h_nb, edge_index, tied_alpha=None):
        N = h_self.shape[0]
        src, dst = edge_index[0], edge_index[1]
        loops = torch.arange(N, device=h_self.device)
        src_all = torch.cat([src, loops]); dst_all = torch.cat([dst, loops])
        msg = torch.cat([h_nb[src], h_self], 0)
        if tied_alpha is None:
            score = (msg * self.att_src).sum(-1) + (h_self * self.att_dst).sum(-1)[dst_all]
            alpha = seg_softmax(torch.sigmoid(F.leaky_relu(score, 0.2)), dst_all, num_nodes=N)
        else:
            alpha = tied_alpha
        out = torch.zeros_like(h_self).index_add_(0, dst_all, msg * alpha.unsqueeze(-1))
        return out, alpha


class MaskedGraphEncoder(nn.Module):
    """Linear G->hidden with graph attention, ELU, Linear hidden->embed = z. The decoder uses the transposed weights and
    the encoder's attention (tied)."""

    def __init__(self, n_genes: int, n_sections: int, hidden: int = 512, embed: int = 30):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_genes, hidden)); self.w2 = nn.Parameter(torch.empty(hidden, embed))
        nn.init.xavier_uniform_(self.w1); nn.init.xavier_uniform_(self.w2)
        self.gat = GATLayer(hidden)
        self.section_emb = nn.Embedding(n_sections, embed); nn.init.normal_(self.section_emb.weight, std=0.02)

    def encode(self, x_masked, x_full, edge_index):
        h, alpha = self.gat(x_masked @ self.w1, x_full @ self.w1, edge_index)
        return F.elu(h) @ self.w2, alpha

    def decode(self, z_dec, edge_index, alpha):
        h3 = z_dec @ self.w2.t()
        h3, _ = self.gat(h3, h3, edge_index, tied_alpha=alpha)
        return F.elu(h3) @ self.w1.t()


def allowed_masks(edge_index, N: int, device):
    """Dense attention masks of the spatial transformer: graph neighbours plus self, and graph neighbours only."""
    A = torch.zeros(N, N, dtype=torch.bool, device=device)
    A[edge_index[0], edge_index[1]] = True; A[edge_index[1], edge_index[0]] = True
    A.fill_diagonal_(False)
    return A | torch.eye(N, dtype=torch.bool, device=device), A


class SpatialBlock(nn.Module):
    """Spatial transformer on z: Laplacian positional encoding, graph-masked multi-head self-attention, feed-forward,
    and a residual projection of z. Returns s' and the self-excluded pass (kept so the random-number stream matches the
    reference implementation)."""

    def __init__(self, embed: int = 30, width: int = 128, heads: int = 4, pe_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.heads = heads; self.width = width
        self.in_proj = nn.Linear(embed, width); self.pe_proj = nn.Linear(pe_dim, width)
        self.ln_q, self.ln_kv, self.ln2, self.ln = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.kv, self.proj = nn.Linear(width, width), nn.Linear(width, 2 * width), nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(width, embed); self.skip = nn.Linear(embed, embed)

    def _pass(self, s, pos, allowed, exclude_self: bool):
        N, W, H = s.shape[0], self.width, self.heads
        x = s + pos; qin = pos if exclude_self else x
        q = self.q(self.ln_q(qin)).view(N, H, W // H).transpose(0, 1)
        k, v = self.kv(self.ln_kv(x)).view(N, 2, H, W // H).permute(1, 2, 0, 3)
        empty = ~allowed.any(1); safe = allowed.clone(); idx = empty.nonzero().squeeze(1); safe[idx, idx] = True
        bias = torch.zeros(1, N, N, device=s.device, dtype=s.dtype).masked_fill(~safe[None], float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(H, N, N), dropout_p=self.drop.p if self.training else 0.0)
        a = self.drop(self.proj(a.transpose(0, 1).reshape(N, W))).masked_fill(empty[:, None], 0.0)
        u = (pos if exclude_self else x) + a
        u = u + self.drop(self.ff(self.ln2(u)))
        return self.out(self.ln(u))

    def forward(self, z, pe, allowed_self, allowed_nb):
        s = self.in_proj(z); pos = self.pe_proj(pe)
        s_prime = self._pass(s, pos, allowed_self, False) + self.skip(z)
        s_nb = self._pass(s, pos, allowed_nb, True)
        return s_prime, s_nb


class Router(nn.Module):
    """Spatially informed expert routing: logits = Linear(s'), mixed half-and-half with the mean of the neighbours'
    logits, then softmax(logits / T)."""

    def __init__(self, embed: int, K: int, mix: float = 0.5, tau: float = 1.0):
        super().__init__()
        self.lin = nn.Linear(embed, K); self.mix = mix; self.tau = tau

    def forward(self, s, edge_index):
        logits = self.lin(s); N = s.shape[0]
        ones = torch.ones(edge_index.shape[1], device=s.device, dtype=s.dtype)
        deg = torch.zeros(N, device=s.device, dtype=s.dtype).index_add_(0, edge_index[1], ones)
        nb = torch.zeros_like(logits).index_add_(0, edge_index[1], logits[edge_index[0]]) / deg.clamp_min(1).unsqueeze(1)
        has = (deg > 0).unsqueeze(1).to(s.dtype)
        logits = (1 - self.mix * has) * logits + self.mix * has * nb
        return (logits / self.tau).softmax(1), logits


class GeneNetworkBank(nn.Module):
    """K low-rank gene networks W_k = G_k G_k^T / d with zero diagonal, shared by all sections.
    ``forward`` predicts the (masked) smoothed expression of every spot from the other genes through the
    spot's mixture sum_k alpha_ik W_k."""

    def __init__(self, n_genes: int, K: int, d: int = 64):
        super().__init__()
        self.G = nn.Parameter(torch.randn(K, n_genes, d) * 0.02); self.bias = nn.Parameter(torch.zeros(n_genes))
        self.K, self.n_genes, self.d = K, n_genes, d

    def forward(self, xbar_obs, alpha):
        h = torch.einsum("ng,kgd->knd", xbar_obs, self.G)
        hw = h * alpha.t().unsqueeze(-1)
        out = torch.einsum("knd,kgd->ng", hw, self.G) / self.d
        diag = (self.G ** 2).sum(-1) / self.d
        out = out - xbar_obs * (alpha @ diag)                              # remove the own-gene term
        return out + self.bias

    @torch.no_grad()
    def matrix(self, k: int) -> torch.Tensor:
        """W_k (genes x genes, zero diagonal)."""
        W = self.G[k] @ self.G[k].t() / self.d
        return W - torch.diag(torch.diagonal(W))


class MANGOSTNet(nn.Module):
    def __init__(self, n_genes: int, n_sections: int, n_net_genes: int, K: int = 10, d_gene: int = 64, hidden: int = 512,
                 embed: int = 30, width: int = 128, heads: int = 4, pe_dim: int = 16):
        super().__init__()
        self.enc = MaskedGraphEncoder(n_genes, n_sections, hidden, embed)
        self.spatial = SpatialBlock(embed, width, heads, pe_dim)
        self.usage = Router(embed, K)
        self.nets = GeneNetworkBank(n_net_genes, K, d_gene)
        self.K = K

    def forward(self, x_masked, x_full, xbar_obs, edge_index, pe, allowed_self, allowed_nb, section_idx):
        z, att = self.enc.encode(x_masked, x_full, edge_index)
        s, _ = self.spatial(z, pe, allowed_self, allowed_nb)
        x_hat = self.enc.decode(s + self.enc.section_emb(section_idx), edge_index, att)    # reconstruction from s'
        alpha, logits = self.usage(s, edge_index)
        y_hat = self.nets(xbar_obs, alpha)
        return {"z": z, "s": s, "x_hat": x_hat, "alpha": alpha, "alpha_soft": logits.softmax(1), "y_hat": y_hat}
