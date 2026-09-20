"""Command-line entry points for artifact inspection and local scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.local import load_patient_graph, load_trial_graph, summarize_datasets
from .inference.loader import MODEL_ALIASES
from .inference.pipeline import InferencePipeline, rerank_score
from .paths import artifact_root


def _inventory(_: argparse.Namespace) -> None:
    manifest = json.loads((artifact_root() / "artifact-manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest["inventory"], indent=2))


def _data_summary(_: argparse.Namespace) -> None:
    print(json.dumps(summarize_datasets(), indent=2))


def _score(args: argparse.Namespace) -> None:
    patient_graph = load_patient_graph(args.patient_id, args.year)
    inclusion_graph, exclusion_graph = load_trial_graph(args.trial_id, args.trial_source)
    patient_text = (
        args.patient_text_file.read_text(encoding="utf-8") if args.patient_text_file else None
    )
    trial_text = args.trial_text_file.read_text(encoding="utf-8") if args.trial_text_file else None
    pipeline = InferencePipeline(args.model, fold=args.fold, device=args.device)
    result = pipeline.score(
        patient_graph,
        inclusion_graph,
        exclusion_graph,
        patient_text=patient_text,
        trial_text=trial_text,
    )
    result["rerank_score"] = rerank_score(result)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="structct", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="list bundled models and datasets")
    inventory.set_defaults(func=_inventory)

    summary = subparsers.add_parser("data-summary", help="inspect local Parquet datasets")
    summary.set_defaults(func=_data_summary)

    score = subparsers.add_parser("score", help="score one bundled patient/trial graph pair")
    score.add_argument("--model", choices=sorted(MODEL_ALIASES), default="2class-base")
    score.add_argument("--fold", type=int, default=0)
    score.add_argument("--year", type=int, required=True)
    score.add_argument("--patient-id", type=int, required=True)
    score.add_argument("--trial-id", required=True)
    score.add_argument("--trial-source", choices=("reranking", "trial-graphs"), default="reranking")
    score.add_argument("--patient-text-file", type=Path)
    score.add_argument("--trial-text-file", type=Path)
    score.add_argument("--device", default="cpu")
    score.set_defaults(func=_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
