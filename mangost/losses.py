"""Training objective: L = L_spot + L_net + w_gate * L_gate + L_smooth + w_floor * L_use.

* L_spot   masked reconstruction of the expression from s' (masked entries weighted 2)
* L_net    likelihood mixture over the K gene networks: -tau * log sum_k alpha_ik exp(-E_ik / tau),
           where E_ik is the masked-entry error of network k alone on spot i
* L_gate   KL(responsibility || alpha): the router learns which network explains each spot
* L_smooth neighbouring spots use similar networks
* L_use    minimum-usage hinge: only networks whose mean usage falls below 1/(4K) are penalised
"""
from __future__ import annotations

import torch


def masked_recon(x_hat, x, mask, w_masked: float = 2.0):
    w = torch.where(mask, torch.full_like(x, w_masked), torch.ones_like(x))
    return (w * (x_hat - x) ** 2).sum() / w.sum()


def usage_smoothness(alpha, edge_index):
    return ((alpha[edge_index[0]] - alpha[edge_index[1]]) ** 2).sum(1).mean()


def min_usage_hinge(alpha, floor: float | None = None):
    K = alpha.shape[1]; floor = 1.0 / (4 * K) if floor is None else floor
    return torch.relu(floor - alpha.mean(0)).sum()


def expert_errors(bank, xbar, mask):
    """[N, K] masked-entry mean squared error of each network alone."""
    N, K = xbar.shape[0], bank.K; xo = xbar.masked_fill(mask, 0.0); cnt = mask.sum(1).clamp_min(1).to(xbar.dtype)
    cols = []
    for k in range(K):
        a = torch.zeros(N, K, device=xbar.device, dtype=xbar.dtype); a[:, k] = 1.0
        y = bank(xo, a); cols.append((((y - xbar) ** 2) * mask).sum(1) / cnt)
    return torch.stack(cols, 1)


def responsibility_loss(E, alpha, tau: float = 0.1, eps: float = 1e-8):
    """Likelihood mixture over the networks. Returns (loss, responsibilities [N, K], detached)."""
    logit = (alpha + eps).log() - E / tau
    loss = -tau * torch.logsumexp(logit, dim=1).mean()
    return loss, logit.softmax(1).detach()


def gate_cross_entropy(resp, alpha, eps: float = 1e-8):
    """KL(responsibility || routing weight)."""
    return (resp * ((resp + eps).log() - (alpha + eps).log())).sum(1).mean()
