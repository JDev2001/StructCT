from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.preprocessing import (
    NODE_ROLE_PATIENT,
    NODE_ROLE_TRIAL_EXC,
    NODE_ROLE_TRIAL_INC,
)
from .layers import GATLayer
from .match import CrossMatchHead
from .pooling import AttentionPool


class _InputProjection(nn.Module):
    PATIENT_ROLE_ID = 0

    def __init__(
        self,
        text_dim,
        symbolic_dim,
        num_node_types,
        node_type_embed_dim,
        num_node_roles,
        node_role_embed_dim,
        text_proj_dim,
        symbolic_proj_dim,
        h_dim,
        noise_std=0.0,
        text_feature_dropout=0.0,
        node_text_mask_prob=0.0,
        text_feature_scale=1.0,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.symbolic_dim = symbolic_dim
        self.noise_std = noise_std
        self.text_feature_dropout = float(text_feature_dropout)
        self.node_text_mask_prob = float(node_text_mask_prob)
        self.text_feature_scale = float(text_feature_scale)
        self.text_proj = nn.Linear(text_dim, text_proj_dim)
        self.sym_proj_patient = nn.Linear(symbolic_dim, symbolic_proj_dim, bias=False)
        self.sym_proj_trial = nn.Linear(symbolic_dim, symbolic_proj_dim, bias=False)
        self.type_embed = nn.Embedding(num_node_types, node_type_embed_dim)
        self.node_role_embed = nn.Embedding(num_node_roles, node_role_embed_dim)
        in_dim = text_proj_dim + symbolic_proj_dim + node_type_embed_dim + node_role_embed_dim
        self.out = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
            nn.LayerNorm(h_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        node_role_id: torch.Tensor,
    ) -> torch.Tensor:
        text = x[..., : self.text_dim]
        sym = x[..., self.text_dim : self.text_dim + self.symbolic_dim]
        is_neg = sym[..., -1:]
        text = text * (1.0 - 2.0 * is_neg)
        if self.training and self.noise_std > 0.0:
            text = text + torch.randn_like(text) * self.noise_std
        if self.training and self.text_feature_dropout > 0.0:
            text = F.dropout(text, p=self.text_feature_dropout, training=True)
        if self.training and self.node_text_mask_prob > 0.0:
            keep = torch.rand(text.shape[0], 1, device=text.device, dtype=text.dtype)
            keep = (keep >= self.node_text_mask_prob).to(text.dtype)
            text = text * keep
        if self.text_feature_scale != 1.0:
            text = text * self.text_feature_scale

        sym_p = self.sym_proj_patient(sym)
        sym_t = self.sym_proj_trial(sym)
        is_patient = (node_role_id == self.PATIENT_ROLE_ID).unsqueeze(-1)
        sym_out = torch.where(is_patient, sym_p, sym_t)

        return self.out(
            torch.cat(
                [
                    self.text_proj(text),
                    sym_out,
                    self.type_embed(node_type),
                    self.node_role_embed(node_role_id),
                ],
                dim=-1,
            )
        )


class EntailmentHead(nn.Module):
    def __init__(self, h_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(4 * h_dim, num_classes)

    def forward(self, h_inc: torch.Tensor, h_exc: torch.Tensor) -> torch.Tensor:
        return self.linear(interaction_features(h_inc, h_exc))


def interaction_features(h_inc: torch.Tensor, h_exc: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [h_inc, h_exc, h_inc * h_exc, (h_inc - h_exc).abs()],
        dim=-1,
    )


def dense_sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).expand_as(z)
    bound = 1 + k * z_sorted > z_cumsum
    k_z = bound.sum(dim=dim, keepdim=True)
    tau = (z_cumsum.gather(dim, (k_z - 1).clamp(min=0)) - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


class RoleAttentionPoolSparsemax(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(h_dim) * 0.02)
        self.v_proj = nn.Linear(h_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)

    def forward(
        self,
        h: torch.Tensor,
        batch_idx: torch.Tensor,
        node_role_id: torch.Tensor,
        pool_mask: torch.Tensor,
        target_role: int,
        batch_size: int,
    ) -> torch.Tensor:
        device = h.device
        keep = pool_mask & (node_role_id == target_role)
        out = h.new_zeros(batch_size, h.shape[-1])
        if not keep.any():
            return out

        hk = h[keep]
        bk = batch_idx[keep]

        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        counts.index_add_(0, bk, torch.ones_like(bk))
        max_n = int(counts.max().item())
        if max_n == 0:
            return out

        order = torch.argsort(bk, stable=True)
        bk_sorted = bk[order]
        hk_sorted = hk[order]

        pos = torch.arange(bk_sorted.shape[0], device=device)
        start = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            start[1:] = counts.cumsum(0)[:-1]
        local_pos = pos - start[bk_sorted]

        H_pad = torch.zeros(batch_size, max_n, h.shape[-1], device=device)
        H_pad[bk_sorted, local_pos] = hk_sorted

        mask = torch.ones(batch_size, max_n, dtype=torch.bool, device=device)
        mask[bk_sorted, local_pos] = False

        K = self.k_proj(H_pad)
        scores = (K * self.query).sum(dim=-1) / (h.shape[-1] ** 0.5)
        scores[mask] = float("-inf")

        attn = dense_sparsemax(scores, dim=-1)

        V = self.v_proj(H_pad)
        return (V * attn.unsqueeze(-1)).sum(dim=1)


class CrossGATModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        text_dim = config.EMBED_DIM - config.SYMBOLIC_STATUS_DIM

        self.input_proj = _InputProjection(
            text_dim=text_dim,
            symbolic_dim=config.SYMBOLIC_STATUS_DIM,
            num_node_types=config.NUM_NODE_TYPES,
            node_type_embed_dim=config.NODE_TYPE_EMBED_DIM,
            num_node_roles=config.NUM_NODE_ROLES,
            node_role_embed_dim=config.NODE_ROLE_EMBED_DIM,
            text_proj_dim=config.TEXT_PROJ_DIM,
            symbolic_proj_dim=config.SYMBOLIC_PROJ_DIM,
            h_dim=config.H_DIM,
            noise_std=config.EMBEDDING_NOISE_STD,
            text_feature_dropout=config.TEXT_FEATURE_DROPOUT,
            node_text_mask_prob=config.NODE_TEXT_MASK_PROB,
            text_feature_scale=config.TEXT_FEATURE_SCALE,
        )

        layer_kwargs = dict(
            h_dim=config.H_DIM,
            num_heads=config.HEADS,
            dropout=config.DROPOUT,
            attn_dropout=config.ATTN_DROPOUT,
            num_edge_types=config.NUM_EDGE_TYPES,
            edge_type_dim=config.EDGE_TYPE_EMBED_DIM,
        )
        self.layers = nn.ModuleList([GATLayer(**layer_kwargs) for _ in range(config.NUM_LAYERS)])

        self.use_cross_attn_head = bool(config.USE_CROSS_ATTN_HEAD)
        self.use_attn_pooling = bool(config.USE_ATTN_POOLING)

        if self.use_attn_pooling:
            self.role_pool = RoleAttentionPoolSparsemax(config.H_DIM)

        self.head = EntailmentHead(config.H_DIM, config.NUM_CLASSES)
        if self.use_cross_attn_head:
            self.cross_match = CrossMatchHead(
                h_dim=config.H_DIM,
                num_heads=int(config.CROSS_ATTN_HEADS),
                hidden_dim=config.MLP_HIDDEN_DIM,
                out_dim=config.H_DIM,
                dropout=config.DROPOUT,
            )
            self.pool = None
        else:
            self.pool = AttentionPool(config.H_DIM, config.READOUT_HEADS, config.DROPOUT)
            self.cross_match = None

    def _encode_nodes(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_role_id = batch.node_role_id
        pool_mask = batch.pool_mask

        h = self.input_proj(batch.x, batch.node_type, node_role_id)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_type, batch.batch)
        return h, node_role_id, pool_mask

    def _pool_global(self, batch, h: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        emb, _ = self.pool(h, batch.batch, pool_mask, int(batch.num_graphs))
        return emb

    @staticmethod
    def _mean_pool_role(
        h: torch.Tensor,
        batch_idx: torch.Tensor,
        node_role_id: torch.Tensor,
        pool_mask: torch.Tensor,
        target_role: int,
        batch_size: int,
    ) -> torch.Tensor:
        keep = pool_mask & (node_role_id == target_role)
        out = h.new_zeros(batch_size, h.shape[-1])
        if not keep.any():
            return out
        cnt = h.new_zeros(batch_size)
        out.index_add_(0, batch_idx[keep], h[keep])
        cnt.index_add_(0, batch_idx[keep], h.new_ones(int(keep.sum().item())))
        return out / cnt.clamp(min=1.0).unsqueeze(-1)

    def _pool_patient(self, h, batch, role, pool_mask, B):
        if self.use_attn_pooling:
            return self.role_pool(h, batch.batch, role, pool_mask, NODE_ROLE_PATIENT, B)
        return self._mean_pool_role(h, batch.batch, role, pool_mask, NODE_ROLE_PATIENT, B)

    def forward(self, inc_batch, exc_batch) -> dict:
        h_inc, role_inc, pool_inc = self._encode_nodes(inc_batch)
        h_exc, role_exc, pool_exc = self._encode_nodes(exc_batch)
        B = int(inc_batch.num_graphs)

        if self.use_cross_attn_head:
            inc_emb = self.cross_match(
                h_inc,
                inc_batch.batch,
                role_inc,
                pool_inc,
                B,
                patient_role_id=NODE_ROLE_PATIENT,
            )
            exc_emb = self.cross_match(
                h_exc,
                exc_batch.batch,
                role_exc,
                pool_exc,
                B,
                patient_role_id=NODE_ROLE_PATIENT,
            )
        else:
            inc_emb = self._pool_global(inc_batch, h_inc, pool_inc)
            exc_emb = self._pool_global(exc_batch, h_exc, pool_exc)

        logits = self.head(inc_emb, exc_emb)

        pat_from_inc = self._pool_patient(h_inc, inc_batch, role_inc, pool_inc, B)
        pat_from_exc = self._pool_patient(h_exc, exc_batch, role_exc, pool_exc, B)
        trial_inc_emb = self._mean_pool_role(
            h_inc, inc_batch.batch, role_inc, pool_inc, NODE_ROLE_TRIAL_INC, B
        )
        trial_exc_emb = self._mean_pool_role(
            h_exc, exc_batch.batch, role_exc, pool_exc, NODE_ROLE_TRIAL_EXC, B
        )

        return {
            "logits": logits,
            "inc_emb_trial": trial_inc_emb,
            "inc_emb_patient": pat_from_inc,
            "exc_emb_trial": trial_exc_emb,
            "exc_emb_patient": pat_from_exc,
            "graph_features": interaction_features(inc_emb, exc_emb),
        }
