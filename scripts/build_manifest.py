"""Build the public artifact inventory and SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".cache", ".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
EXCLUDED_FILES = {
    "artifact-manifest.json",
    "checkpoint_manifest.json",
    "crossgat_hf.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        not any(part in EXCLUDED_DIRS for part in relative.parts)
        and path.name not in EXCLUDED_FILES
        and path.suffix != ".ckpt"
    )


def main() -> None:
    files = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file() and included(item)):
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "format_version": 1,
        "artifact": "StructCT",
        "inventory": {
            "models": [
                {"alias": "full", "classes": 3, "checkpoints": 1},
                {"alias": "cv", "classes": 3, "checkpoints": 5},
                {"alias": "3class-ce", "classes": 3, "checkpoints": 5},
                {"alias": "2class-base", "classes": 2, "checkpoints": 5},
                {"alias": "2class-eligloss", "classes": 2, "checkpoints": 5},
                {"alias": "2class-rank", "classes": 2, "checkpoints": 5},
            ],
            "datasets": [
                {"name": "patient-graphs", "rows": 165},
                {"name": "trial-graphs", "rows": 103878},
                {"name": "reranking", "rows": 13229},
            ],
        },
        "files": files,
        "totals": {"files": len(files), "bytes": sum(item["bytes"] for item in files)},
    }
    destination = ROOT / "artifact-manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination.name}: {manifest['totals']}")


if __name__ == "__main__":
    main()
