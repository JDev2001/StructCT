from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GraphNorm


class GATLayer(nn.Module):
    def __init__(
        self,
        h_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_edge_types: int = 9,
        edge_type_dim: int = 16,
    ):
        super().__init__()
        self.num_edge_types = int(num_edge_types)
        self.edge_type_embed = nn.Embedding(self.num_edge_types * 2, edge_type_dim)
        self.gnorm = GraphNorm(h_dim)
        self.gat = GATv2Conv(
            h_dim,
            h_dim // num_heads,
            heads=num_heads,
            dropout=attn_dropout,
            concat=True,
            edge_dim=edge_type_dim,
        )
        self.fnorm = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, h_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        if h.shape[0] == 0:
            return h
        if edge_index.shape[1] > 0:
            rev = edge_index.flip(0)
            edge_index = torch.cat([edge_index, rev], dim=1)
            rev_types = edge_type + self.num_edge_types
            edge_type = torch.cat([edge_type, rev_types], dim=0)
        edge_attr = self.edge_type_embed(edge_type)
        h = h + self.gat(self.gnorm(h, batch_idx), edge_index, edge_attr=edge_attr)
        h = h + self.ffn(self.fnorm(h))
        return h
