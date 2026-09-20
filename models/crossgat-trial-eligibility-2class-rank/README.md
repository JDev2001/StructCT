---
license: cc-by-nc-4.0
language: [en]
tags:
  - clinical-trials
  - information-retrieval
  - reranking
  - graph-neural-network
  - biomedical
library_name: pytorch
pipeline_tag: text-classification
---

# CrossGAT Stage-B — LambdaRank + retrieval negatives

Two-class eligibility model (excluded vs eligible) trained with a LambdaRank objective and retrieval-distributed negatives.

Output classes: `excluded / eligible`. These are the **five cross-validation checkpoints** —
one per fold, each trained on 80% of the topics and never shown its own fold's.

## Local inputs

This checkpoint consumes parsed patient and trial eligibility graphs. All StructCT checkpoints, graph data, and inference code are bundled locally; none is fetched from the original model or dataset repositories. It is graph-only and does not consume raw patient or trial text.

## What it is for

**It reranks, it does not retrieve.** Trained on an eligibility target, it reorders
*within* the candidate set a first-stage retriever already found. It cannot judge
whether a trial is even on the right disease — per-topic AUC for relevant-vs-rest
is 0.533, i.e. chance. **Fuse it with the retrieval score, never replace it:** the
pure-model ranking (w=1.0 below) is consistently worse than the first stage.

## Benchmark

Fold-consistent 5-fold CV over 162 TREC CT 2021-23 topics, hybrid BM25 + embeddinggemma-300m-medical first stage, top-100 candidates, scored by `p_eligible` and fused convexly with the retrieval score. p from a paired Wilcoxon signed-rank test against the first stage (`zero_method='zsplit'`, so topics left unchanged count as evidence of no effect).

| configuration | graded NDCG@10 | Δ vs first stage | p | judged@10 |
|---|---|---|---|---|
| first stage only | 0.5973 | — | — | 0.922 |
| fusion w=0.3 | 0.6195 | +0.0221 | 0.00335 | 0.911 |
| fusion w=0.5 | 0.6257 | +0.0284 | 0.0068 | 0.878 |
| fusion w=0.7 | 0.5969 | -0.0004 | 0.841 | 0.809 |
| fusion w=1 | 0.5394 | -0.0580 | 0.0138 | 0.730 |

Graded gain is `2^label - 1` (0/1/3). Binary NDCG is not the metric to read here:
excluded and eligible both count as relevant under it, so the ordering this model
is trained to produce is invisible to it by construction.

## Local usage

```python
from structct import InferencePipeline
from structct.data import load_patient_graph, load_trial_graph

patient_graph = load_patient_graph(patient_id=1, year=2021)
inclusion_graph, exclusion_graph = load_trial_graph("NCT00000106")
pipeline = InferencePipeline("2class-rank", fold=0)
result = pipeline.score(
    patient_graph,
    inclusion_graph,
    exclusion_graph,
)
print(result["class_probs"])
```

See the repository-level README for setup, CLI usage, and input requirements.

## Checkpoint provenance

Each `.pt` file contains tensors only. Its adjacent JSON sidecar records the
inference configuration, class order, architecture type, and full SHA-256 digest.

## Related

- [crossgat-trial-eligibility-cv](../crossgat-trial-eligibility-cv/README.md) — 3-class LambdaRank + hard negatives, the best reranker of this family
- [crossgat-trial-eligibility-full](../crossgat-trial-eligibility-full/README.md) — post-CV refit for deployment
