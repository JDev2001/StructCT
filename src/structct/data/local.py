"""Read the bundled datasets directly from local Parquet files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

from ..paths import data_dir
from .schemas import RawGraph

DATASETS = {
    "patient-graphs": "patient-graphs/data",
    "trial-graphs": "trial-graphs/data",
    "reranking": "reranking/data",
}


def _dataset(name: str) -> ds.Dataset:
    try:
        path = data_dir() / DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}") from exc
    return ds.dataset(path, format="parquet")


def _json_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value else []
    return list(value)


def _one_row(table, description: str) -> dict[str, Any]:
    rows = table.to_pylist()
    if not rows:
        raise KeyError(f"no row found for {description}")
    if len(rows) > 1:
        raise ValueError(f"expected one row for {description}, found {len(rows)}")
    return rows[0]


def load_patient_graph(patient_id: int, year: int) -> RawGraph:
    dataset = _dataset("patient-graphs")
    filt = (ds.field("patient_id") == patient_id) & (ds.field("year") == year)
    row = _one_row(dataset.to_table(filter=filt), f"patient {patient_id} in {year}")
    return RawGraph(entities=_json_list(row["entities"]), relations=_json_list(row["relations"]))


def load_trial_graph(nct_id: str, source: str = "reranking") -> tuple[RawGraph, RawGraph]:
    """Return inclusion and exclusion graphs for a trial."""
    dataset = _dataset(source)
    rows = dataset.to_table(filter=ds.field("nct_id") == nct_id).to_pylist()
    if not rows:
        raise KeyError(f"trial {nct_id!r} is absent from {source!r}")

    if source == "reranking":
        row = rows[0]
        return (
            RawGraph(
                entities=_json_list(row["inclusion_entities"]),
                relations=_json_list(row["inclusion_relations"]),
            ),
            RawGraph(
                entities=_json_list(row["exclusion_entities"]),
                relations=_json_list(row["exclusion_relations"]),
            ),
        )

    by_type = {str(row["type"]).lower(): row for row in rows}
    graphs = []
    for graph_type in ("inclusion", "exclusion"):
        row = by_type.get(graph_type)
        graphs.append(
            RawGraph(
                entities=_json_list(row["entities"]) if row else [],
                relations=_json_list(row["relations"]) if row else [],
            )
        )
    return graphs[0], graphs[1]


def summarize_datasets() -> list[dict[str, Any]]:
    summaries = []
    for name, relative_path in DATASETS.items():
        dataset = _dataset(name)
        files = sorted(Path(path) for path in dataset.files)
        summaries.append(
            {
                "name": name,
                "rows": dataset.count_rows(),
                "columns": dataset.schema.names,
                "files": [path.name for path in files],
                "bytes": sum(path.stat().st_size for path in files),
                "path": relative_path,
            }
        )
    return summaries
