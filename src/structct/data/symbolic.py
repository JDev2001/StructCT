from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .embeddings import EmbeddingCache
from .schemas import Entity, Expression

logger = logging.getLogger(__name__)

STATUS_SATISFIED = 0
STATUS_UNSATISFIED = 1
STATUS_UNKNOWN = 2


def _eval_operator(op: str, patient_value: float, trial_value: float) -> bool:
    if op == "<":
        return patient_value < trial_value
    if op == ">":
        return patient_value > trial_value
    if op == "<=":
        return patient_value <= trial_value
    if op == ">=":
        return patient_value >= trial_value
    if op == "==":
        return patient_value == trial_value
    if op == "!=":
        return patient_value != trial_value
    return False


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _expr_subject(expr: Expression) -> str:
    return (expr.subject_name or "").strip()


def evaluate_trial_entity_status(
    trial_entity: Entity,
    patient_entities: list[Entity],
    embedder: EmbeddingCache,
    match_threshold: float = 0.6,
) -> int:
    if not trial_entity.code_expr:
        return STATUS_UNKNOWN

    trial_exprs = [e for e in trial_entity.code_expr if _expr_subject(e)]
    if not trial_exprs:
        return STATUS_UNKNOWN

    patient_exprs: list[tuple[Entity, Expression]] = []
    for pe in patient_entities:
        if not pe.code_expr:
            continue
        for ex in pe.code_expr:
            if _expr_subject(ex):
                patient_exprs.append((pe, ex))
    if not patient_exprs:
        return STATUS_UNKNOWN

    all_subjects = list(
        {_expr_subject(e) for e in trial_exprs} | {_expr_subject(ex) for _, ex in patient_exprs}
    )
    vec_map = embedder.encode_unique(all_subjects)

    any_evaluated = False
    any_unsatisfied = False
    all_satisfied = True

    for texp in trial_exprs:
        ts = _expr_subject(texp)
        if ts not in vec_map:
            continue
        tvec = vec_map[ts]
        for _pe, pexp in patient_exprs:
            ps = _expr_subject(pexp)
            if ps not in vec_map:
                continue
            if _cos(tvec, vec_map[ps]) < match_threshold:
                continue
            any_evaluated = True
            if not _eval_operator(texp.operator, pexp.value, texp.value):
                any_unsatisfied = True
                all_satisfied = False

    if not any_evaluated:
        return STATUS_UNKNOWN
    if any_unsatisfied:
        return STATUS_UNSATISFIED
    if all_satisfied:
        return STATUS_SATISFIED
    return STATUS_UNKNOWN


def status_onehot(status: int) -> torch.Tensor:
    v = torch.zeros(3, dtype=torch.float32)
    v[status] = 1.0
    return v
