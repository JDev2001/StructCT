"""Load sanitized model weights into inference-only PyTorch modules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import Config
from ..model.crossgat_match import CrossGATModel
from ..model.joint_model import JointModel
from ..paths import models_dir

MODEL_ALIASES = {
    "full": "crossgat-trial-eligibility-full",
    "cv": "crossgat-trial-eligibility-cv",
    "2class-base": "crossgat-trial-eligibility-2class-base",
    "2class-eligloss": "crossgat-trial-eligibility-2class-eligloss",
    "2class-rank": "crossgat-trial-eligibility-2class-rank",
    "3class-ce": "crossgat-trial-eligibility-3class-ce",
}


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    config: Config
    classes: tuple[str, ...]
    architecture: str
    checkpoint_path: Path


def resolve_model_dir(model: str | Path) -> Path:
    candidate = Path(model)
    if candidate.is_dir():
        return candidate.resolve()
    name = MODEL_ALIASES.get(str(model), str(model))
    path = models_dir() / name
    if not path.is_dir():
        raise FileNotFoundError(f"unknown model {model!r}; aliases: {sorted(MODEL_ALIASES)}")
    return path


def resolve_checkpoint(model_dir: Path, fold: int | None) -> Path:
    if fold is None:
        path = model_dir / "checkpoints" / "model.pt"
    else:
        if fold not in range(5):
            raise ValueError("fold must be between 0 and 4")
        path = model_dir / "checkpoints" / f"fold{fold}" / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_model(
    model: str | Path,
    *,
    fold: int | None = None,
    device: str = "cpu",
) -> LoadedModel:
    model_dir = resolve_model_dir(model)
    checkpoint = resolve_checkpoint(model_dir, fold)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config = Config.from_dict(metadata["config"])

    if metadata["architecture"] == "joint":
        module: torch.nn.Module = JointModel(config, desc_embed_dim=config.DESC_EMBED_DIM)
    elif metadata["architecture"] == "crossgat":
        module = CrossGATModel(config)
    else:
        raise ValueError(f"unsupported architecture {metadata['architecture']!r}")

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()
    return LoadedModel(
        model=module,
        config=config,
        classes=tuple(metadata["classes"]),
        architecture=metadata["architecture"],
        checkpoint_path=checkpoint,
    )
