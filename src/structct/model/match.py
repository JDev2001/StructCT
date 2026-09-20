from __future__ import annotations

import torch
import torch.nn as nn


class CrossMatchHead(nn.Module):
    def __init__(self, h_dim: int, num_heads: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.h_dim = h_dim
        self.attn_p2t = nn.MultiheadAttention(h_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_t2p = nn.MultiheadAttention(h_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(h_dim)
        self.norm_t = nn.LayerNorm(h_dim)
        self.null_p = nn.Parameter(torch.randn(h_dim) * 0.02)
        self.null_t = nn.Parameter(torch.randn(h_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(4 * h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    @staticmethod
    def _pad_per_graph(
        h: torch.Tensor,
        batch_idx: torch.Tensor,
        keep: torch.Tensor,
        batch_size: int,
        h_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = h.device
        if keep.any():
            hk = h[keep]
            bk = batch_idx[keep]
        else:
            return (
                torch.zeros(batch_size, 1, h_dim, device=device),
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            )
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        counts.index_add_(0, bk, torch.ones_like(bk))
        max_n = int(counts.max().item()) if counts.numel() else 1
        if max_n == 0:
            max_n = 1
        order = torch.argsort(bk, stable=True)
        bk_sorted = bk[order]
        hk_sorted = hk[order]
        pos = torch.arange(bk_sorted.shape[0], device=device)
        start = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            start[1:] = counts.cumsum(0)[:-1]
        local_pos = pos - start[bk_sorted]
        out = torch.zeros(batch_size, max_n, h_dim, device=device)
        out[bk_sorted, local_pos] = hk_sorted
        pad_mask = torch.ones(batch_size, max_n, dtype=torch.bool, device=device)
        pad_mask[bk_sorted, local_pos] = False
        return out, pad_mask

    def forward(
        self,
        h: torch.Tensor,
        batch_idx: torch.Tensor,
        node_role_id: torch.Tensor,
        pool_mask: torch.Tensor,
        batch_size: int,
        patient_role_id: int = 0,
    ) -> torch.Tensor:
        keep_p = pool_mask & (node_role_id == patient_role_id)
        keep_t = pool_mask & (node_role_id != patient_role_id)
        Hp, mask_p = self._pad_per_graph(h, batch_idx, keep_p, batch_size, self.h_dim)
        Ht, mask_t = self._pad_per_graph(h, batch_idx, keep_t, batch_size, self.h_dim)

        empty_p = mask_p.all(dim=1)
        empty_t = mask_t.all(dim=1)
        if empty_p.any():
            Hp[empty_p, 0] = self.null_p
            mask_p = mask_p.clone()
            mask_p[empty_p, 0] = False
        if empty_t.any():
            Ht[empty_t, 0] = self.null_t
            mask_t = mask_t.clone()
            mask_t[empty_t, 0] = False

        Hp2, _ = self.attn_p2t(self.norm_p(Hp), Ht, Ht, key_padding_mask=mask_t, need_weights=False)
        Ht2, _ = self.attn_t2p(self.norm_t(Ht), Hp, Hp, key_padding_mask=mask_p, need_weights=False)

        Hp_all = Hp + Hp2
        Ht_all = Ht + Ht2

        def _mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            valid = (~mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            return (x * valid).sum(dim=1) / denom

        p_vec = _mean(Hp_all, mask_p)
        t_vec = _mean(Ht_all, mask_t)
        feat = torch.cat([p_vec, t_vec, p_vec * t_vec, (p_vec - t_vec).abs()], dim=-1)
        return self.mlp(feat)
