from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .embeddings import EmbeddingCache
from .schemas import Entity, RawGraph, RawSample
from .symbolic import (
    STATUS_UNKNOWN,
    evaluate_trial_entity_status,
    status_onehot,
)

logger = logging.getLogger(__name__)

SYMBOLIC_STATUS_DIM = 4

SEMANTIC_ROOT_TYPES = frozenset(
    {
        "condition",
        "measurement",
        "procedure",
        "drug",
        "person",
        "device",
        "visit",
        "observation",
    }
)

LOGIC_GROUP_TYPES = frozenset({"logicgroup_or", "logicgroup_and"})
POOL_TYPES = SEMANTIC_ROOT_TYPES | LOGIC_GROUP_TYPES

CANONICAL_NODE_TYPES: list[str] = [
    "condition",
    "measurement",
    "procedure",
    "drug",
    "person",
    "device",
    "visit",
    "observation",
    "value",
    "temporal",
    "negation",
    "qualifier",
    "scope",
    "logicgroup_or",
    "logicgroup_and",
    "reference_point",
    "multiplier",
    "mood",
    "other",
]

CANONICAL_RELATION_TYPES: list[str] = [
    "has_value",
    "has_temporal",
    "has_qualifier",
    "has_negation",
    "has_scope",
    "and",
    "or",
    "not",
    "has_context",
    "has_index",
    "subsumes",
    "has_multiplier",
    "has_mood",
    "cross",
    "cross_any",
    "same_any",
    "other",
]

_NODE_TYPE_IDX = {t: i for i, t in enumerate(CANONICAL_NODE_TYPES)}
_REL_TYPE_IDX = {t: i for i, t in enumerate(CANONICAL_RELATION_TYPES)}

_CROSS_TYPE_ID = _REL_TYPE_IDX["cross"]
_CROSS_ANY_TYPE_ID = _REL_TYPE_IDX["cross_any"]
_SAME_ANY_TYPE_ID = _REL_TYPE_IDX["same_any"]

NODE_ROLE_PATIENT = 0
NODE_ROLE_TRIAL_INC = 1
NODE_ROLE_TRIAL_EXC = 2
NUM_NODE_ROLES = 3


def num_node_types() -> int:
    return len(CANONICAL_NODE_TYPES)


def num_relation_types() -> int:
    return len(CANONICAL_RELATION_TYPES)


def _norm_type(t: str) -> str:
    return (t or "").strip().lower()


def _node_type_id(t: str) -> int:
    return _NODE_TYPE_IDX.get(_norm_type(t), _NODE_TYPE_IDX["other"])


def _rel_type_id(t: str) -> int:
    return _REL_TYPE_IDX.get(_norm_type(t), _REL_TYPE_IDX["other"])


def _is_root_type(t: str) -> bool:
    return _norm_type(t) in SEMANTIC_ROOT_TYPES


def _is_pool_type(t: str) -> bool:
    return _norm_type(t) in POOL_TYPES


def _collect_negated_ids(graph: RawGraph) -> set[str]:
    types_by_id = {e.id: _norm_type(e.type) for e in graph.entities}
    out: set[str] = set()
    for r in graph.relations:
        if _norm_type(r.type) != "has_negation":
            continue
        a_neg = types_by_id.get(r.arg1_id) == "negation"
        b_neg = types_by_id.get(r.arg2_id) == "negation"
        if a_neg and not b_neg:
            out.add(r.arg2_id)
        elif b_neg and not a_neg:
            out.add(r.arg1_id)
        else:
            out.add(r.arg1_id)
    return out


def _build_edges(
    graph: RawGraph,
    id_to_gidx: dict[str, int],
) -> tuple[list[int], list[int], list[int]]:
    src, dst, etype = [], [], []
    for r in graph.relations:
        a = id_to_gidx.get(r.arg1_id)
        b = id_to_gidx.get(r.arg2_id)
        if a is None or b is None or a == b:
            continue
        src.append(a)
        dst.append(b)
        etype.append(_rel_type_id(r.type))
    return src, dst, etype


def build_combined_graph(
    patient_graph: RawGraph,
    trial_graph: RawGraph,
    embedder: EmbeddingCache,
    patient_entities: list[Entity] | None = None,
    symbolic_match_threshold: float = 0.6,
    trial_role: int = NODE_ROLE_TRIAL_INC,
    use_same_any: bool = True,
    use_dense_cross_any: bool = True,
) -> Data:
    P = len(patient_graph.entities)
    T = len(trial_graph.entities)
    dim = embedder.dim

    if P == 0 and T == 0:
        empty_x = torch.zeros((0, dim + SYMBOLIC_STATUS_DIM), dtype=torch.float32)
        return Data(
            x=empty_x,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            node_type=torch.zeros((0,), dtype=torch.long),
            root_mask=torch.zeros((0,), dtype=torch.bool),
            pool_mask=torch.zeros((0,), dtype=torch.bool),
            node_role_id=torch.zeros((0,), dtype=torch.long),
        )

    all_texts = [e.text or "" for e in patient_graph.entities] + [
        e.text or "" for e in trial_graph.entities
    ]
    all_emb = embedder.encode(all_texts)

    if T > 0:
        trial_statuses = [
            evaluate_trial_entity_status(
                e, patient_entities or [], embedder, symbolic_match_threshold
            )
            for e in trial_graph.entities
        ]
        trial_sym = torch.stack([status_onehot(s) for s in trial_statuses])
    else:
        trial_statuses = []
        trial_sym = torch.zeros((0, 3), dtype=torch.float32)

    trial_root_type_best: dict[str, int] = {}
    for i, e in enumerate(trial_graph.entities):
        t = _norm_type(e.type)
        if _is_root_type(t):
            s = trial_statuses[i]
            if t not in trial_root_type_best or s < trial_root_type_best[t]:
                trial_root_type_best[t] = s

    if P > 0:
        patient_sym = torch.stack(
            [
                status_onehot(
                    trial_root_type_best.get(_norm_type(e.type), STATUS_UNKNOWN)
                    if _is_root_type(_norm_type(e.type))
                    else STATUS_UNKNOWN
                )
                for e in patient_graph.entities
            ]
        )
    else:
        patient_sym = torch.zeros((0, 3), dtype=torch.float32)

    sym_onehot = torch.cat([patient_sym, trial_sym], dim=0)

    p_negated = _collect_negated_ids(patient_graph)
    t_negated = _collect_negated_ids(trial_graph)
    p_neg_vec = (
        torch.tensor(
            [1.0 if e.id in p_negated else 0.0 for e in patient_graph.entities],
            dtype=torch.float32,
        ).unsqueeze(-1)
        if P > 0
        else torch.zeros((0, 1), dtype=torch.float32)
    )
    t_neg_vec = (
        torch.tensor(
            [1.0 if e.id in t_negated else 0.0 for e in trial_graph.entities],
            dtype=torch.float32,
        ).unsqueeze(-1)
        if T > 0
        else torch.zeros((0, 1), dtype=torch.float32)
    )
    is_negated = torch.cat([p_neg_vec, t_neg_vec], dim=0)

    x = torch.cat([all_emb, sym_onehot, is_negated], dim=-1)

    p_types = [_norm_type(e.type) for e in patient_graph.entities]
    t_types = [_norm_type(e.type) for e in trial_graph.entities]

    node_type = torch.tensor(
        [_node_type_id(t) for t in p_types] + [_node_type_id(t) for t in t_types],
        dtype=torch.long,
    )

    root_mask = torch.tensor(
        [_is_root_type(t) for t in p_types] + [_is_root_type(t) for t in t_types],
        dtype=torch.bool,
    )
    pool_mask = torch.tensor(
        [_is_pool_type(t) for t in p_types] + [_is_pool_type(t) for t in t_types],
        dtype=torch.bool,
    )

    node_role_id = torch.tensor(
        [NODE_ROLE_PATIENT] * P + [trial_role] * T,
        dtype=torch.long,
    )

    p_id_to_gidx = {e.id: i for i, e in enumerate(patient_graph.entities)}
    t_id_to_gidx = {e.id: P + i for i, e in enumerate(trial_graph.entities)}

    src, dst, etype = _build_edges(patient_graph, p_id_to_gidx)
    ts, td, te = _build_edges(trial_graph, t_id_to_gidx)
    src += ts
    dst += td
    etype += te

    p_root_indices = [i for i, t in enumerate(p_types) if _is_root_type(t)]
    t_root_indices = [P + i for i, t in enumerate(t_types) if _is_root_type(t)]

    if p_root_indices and t_root_indices:
        p_emb_norm = F.normalize(all_emb[p_root_indices], dim=-1)
        t_emb_norm = F.normalize(all_emb[t_root_indices], dim=-1)
        sim_matrix = (p_emb_norm @ t_emb_norm.T).clamp(-1.0, 1.0)
    else:
        sim_matrix = None

    for pi in p_root_indices:
        pt = p_types[pi]
        for ti in t_root_indices:
            tt = t_types[ti - P]
            if pt == tt:
                src.append(pi)
                dst.append(ti)
                etype.append(_CROSS_TYPE_ID)

    if sim_matrix is not None and use_dense_cross_any:
        for pi in p_root_indices:
            for ti in t_root_indices:
                src.append(pi)
                dst.append(ti)
                etype.append(_CROSS_ANY_TYPE_ID)

    if use_same_any:
        p_pool_indices = [i for i, t in enumerate(p_types) if _is_pool_type(t)]
        t_pool_indices = [P + i for i, t in enumerate(t_types) if _is_pool_type(t)]
        for a in p_pool_indices:
            for b in p_pool_indices:
                if a != b:
                    src.append(a)
                    dst.append(b)
                    etype.append(_SAME_ANY_TYPE_ID)
        for a in t_pool_indices:
            for b in t_pool_indices:
                if a != b:
                    src.append(a)
                    dst.append(b)
                    etype.append(_SAME_ANY_TYPE_ID)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        root_mask=root_mask,
        pool_mask=pool_mask,
        node_role_id=node_role_id,
    )


class PreprocessedSample:
    __slots__ = (
        "inc_combined",
        "exc_combined",
        "label",
        "topic_id",
        "year",
        "nct_id",
        "patient_description",
    )

    def __init__(
        self,
        inc_combined: Data,
        exc_combined: Data,
        label: int,
        topic_id: int,
        year: int,
        nct_id: str,
        patient_description: str | None = None,
    ):
        self.inc_combined = inc_combined
        self.exc_combined = exc_combined
        self.label = label
        self.topic_id = topic_id
        self.year = year
        self.nct_id = nct_id
        self.patient_description = patient_description


def preprocess_sample(
    sample: RawSample,
    embedder: EmbeddingCache,
    symbolic_match_threshold: float = 0.6,
    use_same_any: bool = True,
    use_dense_cross_any: bool = True,
) -> PreprocessedSample:
    patient_entities = sample.patient_graph.entities
    topo = dict(use_same_any=use_same_any, use_dense_cross_any=use_dense_cross_any)
    inc_combined = build_combined_graph(
        sample.patient_graph,
        sample.inc_trial_graph,
        embedder,
        patient_entities=patient_entities,
        symbolic_match_threshold=symbolic_match_threshold,
        trial_role=NODE_ROLE_TRIAL_INC,
        **topo,
    )
    exc_combined = build_combined_graph(
        sample.patient_graph,
        sample.exc_trial_graph,
        embedder,
        patient_entities=patient_entities,
        symbolic_match_threshold=symbolic_match_threshold,
        trial_role=NODE_ROLE_TRIAL_EXC,
        **topo,
    )
    return PreprocessedSample(
        inc_combined=inc_combined,
        exc_combined=exc_combined,
        label=sample.label,
        topic_id=sample.topic_id,
        year=sample.year,
        nct_id=sample.nct_id,
        patient_description=sample.patient_description,
    )


def cache_key(
    sbert_name: str,
    match_threshold: float,
    filter_flags: dict,
    topology: dict | None = None,
) -> str:
    payload_obj = {
        "sbert": sbert_name,
        "thr": match_threshold,
        "filters": filter_flags,
        "sym_dim": SYMBOLIC_STATUS_DIM,
        "arch": "combined_graph_v7_patient_status_prop",
        "topology": topology or {},
    }
    return hashlib.sha256(json.dumps(payload_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: str, key: str) -> Path:
    p = Path(cache_dir) / f"preproc_{key}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_preprocessed(samples: list[PreprocessedSample], path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    logger.info("Cached %d preprocessed samples → %s", len(samples), path)


def load_preprocessed(path: Path) -> list[PreprocessedSample]:
    with open(path, "rb") as f:
        out = pickle.load(f)
    logger.info("Loaded %d preprocessed samples from cache %s", len(out), path)
    return out
