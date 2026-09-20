from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as scatter_softmax


class AttentionPool(nn.Module):
    def __init__(self, h_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert h_dim % num_heads == 0
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.head_dim = h_dim // num_heads
        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim) * 0.02)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)
        self.out_proj = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = x.device
        if mask.any():
            xm = x[mask]
            bm = batch_idx[mask]
        else:
            xm = x.new_zeros((0, self.h_dim))
            bm = batch_idx.new_zeros((0,))

        if xm.shape[0] == 0:
            return torch.zeros(batch_size, self.h_dim, device=device), None

        k = self.k_proj(xm).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(xm).view(-1, self.num_heads, self.head_dim)
        q = self.query.to(device)
        scores = (k * q).sum(dim=-1) * self.scale

        attn = torch.stack(
            [
                scatter_softmax(scores[:, h], bm, num_nodes=batch_size)
                for h in range(self.num_heads)
            ],
            dim=-1,
        )
        attn = self.dropout(attn)
        weighted = v * attn.unsqueeze(-1)
        out = torch.zeros(batch_size, self.num_heads, self.head_dim, device=device)
        out = out.index_add_(0, bm, weighted)
        out = out.reshape(batch_size, self.h_dim)
        return self.out_proj(out), attn.mean(dim=-1)
