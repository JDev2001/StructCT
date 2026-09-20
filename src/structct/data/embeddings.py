from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DISK_IO_WORKERS = 32
DEFAULT_CACHE_SIZE = 50000


def _sanitize(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: str | Path = ".cache/embeddings",
        max_memory: int = DEFAULT_CACHE_SIZE,
        batch_size: int = 128,
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.cache_dir = Path(cache_dir) / _sanitize(model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory = max_memory
        self.batch_size = batch_size
        self._mem: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading text encoder %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            vec = self.encode(["probe"])
            self._dim = int(vec.shape[-1])
        return self._dim

    def _disk_path(self, h: str) -> Path:
        return self.cache_dir / h[:2] / f"{h}.pt"

    def _load_from_disk(self, h: str) -> torch.Tensor | None:
        p = self._disk_path(h)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                data = f.read()
                try:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True).float()
        except Exception as e:
            logger.warning("Corrupt cache entry %s: %s", p, e)
        return None

    def _save_to_disk(self, h: str, vec: torch.Tensor) -> None:
        p = self._disk_path(h)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vec.float().cpu(), p)

    def _mem_put(self, h: str, vec: torch.Tensor) -> None:
        self._mem[h] = vec
        self._mem.move_to_end(h)
        while len(self._mem) > self.max_memory:
            self._mem.popitem(last=False)

    def encode(self, texts: list[str]) -> torch.Tensor:
        if len(texts) == 0:
            dim = self._dim or 768
            return torch.zeros((0, dim), dtype=torch.float32)

        hashes = [_text_hash(t or "") for t in texts]
        out: list[torch.Tensor | None] = [None] * len(texts)
        to_encode_idx: list[int] = []
        to_encode_text: list[str] = []

        disk_needed: list[tuple[int, str]] = []
        for i, h in enumerate(hashes):
            if h in self._mem:
                out[i] = self._mem[h]
                self._mem.move_to_end(h)
            else:
                disk_needed.append((i, h))

        if disk_needed:

            def _load(item: tuple[int, str]) -> tuple[int, str, torch.Tensor | None]:
                idx, h = item
                return idx, h, self._load_from_disk(h)

            with ThreadPoolExecutor(max_workers=_DISK_IO_WORKERS) as pool:
                for idx, h, vec in pool.map(_load, disk_needed):
                    if vec is not None:
                        self._mem_put(h, vec)
                        out[idx] = vec
                    else:
                        to_encode_idx.append(idx)
                        to_encode_text.append(texts[idx] or "")

        if to_encode_text:
            with torch.no_grad():
                arr = self.model.encode(
                    to_encode_text,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            arr = arr.detach().cpu().float()
            for local_i, global_i in enumerate(to_encode_idx):
                vec = arr[local_i]
                h = hashes[global_i]
                self._mem_put(h, vec)
                self._save_to_disk(h, vec)
                out[global_i] = vec

        stacked = torch.stack([o for o in out])
        if self._dim is None:
            self._dim = int(stacked.shape[-1])
        return stacked

    def encode_unique(self, texts: list[str]) -> dict[str, torch.Tensor]:
        unique = list({t or "" for t in texts})
        if not unique:
            return {}
        vecs = self.encode(unique)
        return {t: vecs[i] for i, t in enumerate(unique)}
