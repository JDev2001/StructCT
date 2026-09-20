from __future__ import annotations

from dataclasses import fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .crossgat_match import CrossGATModel


class JointModel(nn.Module):
    def __init__(self, config: Config, desc_embed_dim: int):
        super().__init__()
        self.config = config
        self.graph = CrossGATModel(config)

        desc_dim = int(config.DESC_FEATURE_DIM)
        mlp_hidden = int(config.MLP_HIDDEN_DIM)
        dropout = float(config.DROPOUT)
        desc_in_dim = 3 * int(desc_embed_dim) + 1
        self.desc_projector = nn.Sequential(
            nn.Linear(desc_in_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, desc_dim),
            nn.LayerNorm(desc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        graph_dim = 4 * int(config.H_DIM)
        self.final_head = nn.Linear(graph_dim + desc_dim, int(config.NUM_CLASSES))

    def forward(self, batch: dict) -> dict:
        p_emb = batch["patient_emb"]
        t_emb = batch["trial_emb"]
        desc_features = self.desc_projector(_desc_inputs(p_emb, t_emb))
        graph_out = self.graph(batch["inc"], batch["exc"])
        graph_features = graph_out["graph_features"]
        logits = self.final_head(torch.cat([graph_features, desc_features], dim=-1))
        return {
            "logits": logits,
            "desc_features": desc_features,
            "graph_features": graph_features,
            **{f"graph_{k}": v for k, v in graph_out.items() if k != "graph_features"},
        }


def _clone_cfg_2class(config: Config) -> Config:
    cfg2 = Config(**{f.name: getattr(config, f.name) for f in fields(config)})
    cfg2.NUM_CLASSES = 2
    return cfg2


def _desc_inputs(patient_emb: torch.Tensor, trial_emb: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(patient_emb, trial_emb, dim=-1, eps=1e-8).unsqueeze(-1)
    return torch.cat([patient_emb, trial_emb, (patient_emb - trial_emb).abs(), cos], dim=-1)
