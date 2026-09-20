"""Repository-relative artifact paths."""

from __future__ import annotations

import os
from pathlib import Path


def artifact_root() -> Path:
    """Return the artifact root, optionally overridden by ``STRUCTCT_ROOT``."""
    override = os.environ.get("STRUCTCT_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    return artifact_root() / "models"


def data_dir() -> Path:
    return artifact_root() / "data"
