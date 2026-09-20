"""Minimal, inference-only model configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Config:
    H_DIM: int = 256
    NUM_LAYERS: int = 3
    HEADS: int = 4
    READOUT_HEADS: int = 4
    DROPOUT: float = 0.2
    ATTN_DROPOUT: float = 0.1
    NODE_TYPE_EMBED_DIM: int = 64
    EDGE_TYPE_EMBED_DIM: int = 32
    NODE_ROLE_EMBED_DIM: int = 16
    NUM_NODE_ROLES: int = 3
    TEXT_PROJ_DIM: int = 256
    SYMBOLIC_PROJ_DIM: int = 64
    SYMBOLIC_STATUS_DIM: int = 4
    MLP_HIDDEN_DIM: int = 256
    NUM_CLASSES: int = 3
    DESC_FEATURE_DIM: int = 256
    NUM_NODE_TYPES: int = 19
    NUM_EDGE_TYPES: int = 17
    EMBED_DIM: int = 772
    DESC_EMBED_DIM: int = 768
    EMBEDDING_NOISE_STD: float = 0.0
    TEXT_FEATURE_DROPOUT: float = 0.0
    NODE_TEXT_MASK_PROB: float = 0.0
    TEXT_FEATURE_SCALE: float = 1.0
    SYMBOLIC_MATCH_THRESHOLD: float = 0.8
    USE_SAME_ANY: bool = False
    USE_DENSE_CROSS_ANY: bool = True
    USE_CROSS_ATTN_HEAD: bool = False
    USE_ATTN_POOLING: bool = False
    CROSS_ATTN_HEADS: int = 4

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Config:
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
