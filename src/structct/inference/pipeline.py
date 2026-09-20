"""End-to-end local graph scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from ..data.embeddings import EmbeddingCache
from ..data.preprocessing import NODE_ROLE_TRIAL_EXC, NODE_ROLE_TRIAL_INC, build_combined_graph
from ..data.schemas import RawGraph
from ..paths import artifact_root
from .loader import LoadedModel, load_model


class InferencePipeline:
    """Load one bundled checkpoint and score patient/trial graph pairs offline."""

    def __init__(
        self,
        model: str | Path,
        *,
        fold: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.loaded: LoadedModel = load_model(model, fold=fold, device=device)
        self.config = self.loaded.config
        self.embedder = EmbeddingCache(
            model_name="abhinand/MedEmbed-base-v0.1",
            device=device,
            cache_dir=artifact_root() / ".cache" / "embeddings",
        )

    def score(
        self,
        patient_graph: RawGraph,
        inclusion_graph: RawGraph,
        exclusion_graph: RawGraph,
        *,
        patient_text: str | None = None,
        trial_text: str | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        topology = {
            "symbolic_match_threshold": cfg.SYMBOLIC_MATCH_THRESHOLD,
            "use_same_any": cfg.USE_SAME_ANY,
            "use_dense_cross_any": cfg.USE_DENSE_CROSS_ANY,
        }
        inc_data = build_combined_graph(
            patient_graph,
            inclusion_graph,
            self.embedder,
            patient_entities=patient_graph.entities,
            trial_role=NODE_ROLE_TRIAL_INC,
            **topology,
        )
        exc_data = build_combined_graph(
            patient_graph,
            exclusion_graph,
            self.embedder,
            patient_entities=patient_graph.entities,
            trial_role=NODE_ROLE_TRIAL_EXC,
            **topology,
        )
        inc_batch = Batch.from_data_list([inc_data]).to(self.device)
        exc_batch = Batch.from_data_list([exc_data]).to(self.device)

        with torch.inference_mode():
            if self.loaded.architecture == "joint":
                if not patient_text or not trial_text:
                    raise ValueError(
                        "three-class checkpoints require patient_text and trial_text; "
                        "the TREC texts are not redistributed in this artifact"
                    )
                descriptions = self.embedder.encode([patient_text, trial_text]).to(self.device)
                output = self.loaded.model(
                    {
                        "inc": inc_batch,
                        "exc": exc_batch,
                        "patient_emb": descriptions[0].unsqueeze(0),
                        "trial_emb": descriptions[1].unsqueeze(0),
                    }
                )
            else:
                output = self.loaded.model(inc_batch, exc_batch)

        logits = output["logits"].squeeze(0).float().cpu()
        probabilities = F.softmax(logits, dim=-1).tolist()
        predicted = int(torch.tensor(probabilities).argmax().item())
        return {
            "label": self.loaded.classes[predicted],
            "pred": predicted,
            "probs": probabilities,
            "class_probs": dict(zip(self.loaded.classes, probabilities, strict=True)),
            "logits": logits.tolist(),
        }


def rerank_score(result: dict[str, Any], method: str = "p_eligible") -> float:
    probabilities = result["class_probs"]
    if method == "p_eligible":
        return float(probabilities["eligible"])
    if method == "one_minus_irrelevant":
        return 1.0 - float(probabilities.get("irrelevant", 0.0))
    if method == "expected_gain":
        return float(probabilities.get("excluded", 0.0)) + 3.0 * float(probabilities["eligible"])
    raise ValueError(f"unknown reranking method {method!r}")
