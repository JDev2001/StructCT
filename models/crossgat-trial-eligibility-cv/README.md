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

# CrossGAT Trial Eligibility — 5-fold CV checkpoints

A 3-class eligibility model that **reranks clinical-trial retrieval candidates**.
It scores a patient description against a trial's inclusion/exclusion criteria as
`irrelevant / excluded / eligible`, by fusing a graph encoder over parsed
eligibility criteria with a text branch over whole-document embeddings.

These are the **five cross-validation checkpoints** — one per fold, so each is
trained on 80% of the data and never saw its own fold's topics. Use them to
reproduce the reported numbers, or ensemble them. For a single deployment model,
see [crossgat-trial-eligibility-full](../crossgat-trial-eligibility-full/README.md).

## Local inputs

This checkpoint consumes parsed patient and trial eligibility graphs. All StructCT checkpoints, graph data, and inference code are bundled locally; none is fetched from the original model or dataset repositories. It also requires the original patient description and trial title/summary. Those TREC texts are not redistributed and must be supplied by an authorized reviewer.

## What it is for, and what it is not for

**It reranks. It does not retrieve.** Trained on an eligibility target, it reorders
*within* the candidate set a first-stage retriever already found — promoting
eligible trials over merely topically-relevant ones.

**It cannot judge disease relevance.** Per-topic AUC for relevant-vs-rest is
**0.533** — chance. Whether a trial is even on the right disease is entirely the
first stage's job. This is why it must be *fused* with retrieval, never used to
replace it.

## Recommended configuration

```
score = (1 - w) * retrieval_norm + w * p_eligible_norm      with w = 0.5
```

Both terms min-max normalised **within a topic**. `p_eligible` = P(eligible) alone
wins over the alternatives in all four runs, at every cutoff, in both metric
families — even though `expected_gain` models the graded gain directly and is what
training optimises.

## Results

5-fold cross-validation, TREC Clinical Trials 2021–23, **162 topics**, paired
Wilcoxon signed-rank against each run's own first stage. Metric is
**graded NDCG@10** (gain `2^label - 1` = 0/1/3).

First stages: hybrid BM25+`embeddinggemma-300m-medical` (retrieval 0.5973) and
dense-only `MedEmbed-base-v0.1` (retrieval 0.5187).

| Variant | gemma-med | Δ | medembed | Δ |
|---|---|---|---|---|
| w=0.3 `p_eligible` | 0.6340 | +0.0367 *** | 0.5467 | +0.0280 *** |
| w=0.3 `expected_gain` | 0.6300 | +0.0327 *** | 0.5433 | +0.0246 *** |
| w=0.3 `one_minus_irrelevant` | 0.6230 | +0.0257 *** | 0.5334 | +0.0147 *** |
| w=0.5 `p_eligible` | 0.6515 | +0.0542 *** | 0.5644 | +0.0457 *** |
| w=0.5 `expected_gain` | 0.6485 | +0.0512 *** | 0.5572 | +0.0385 *** |
| w=0.5 `one_minus_irrelevant` | 0.6302 | +0.0329 *** | 0.5377 | +0.0190 ** |
| w=0.7 `p_eligible` | 0.6491 | +0.0518 *** | 0.5548 | +0.0361 *** |
| w=0.7 `expected_gain` | 0.6455 | +0.0482 *** | 0.5536 | +0.0349 *** |
| w=0.7 `one_minus_irrelevant` | 0.6262 | +0.0289 ** | 0.5405 | +0.0218 ** |
| w=1 `p_eligible` | 0.6191 | +0.0218 n.s. | 0.5202 | +0.0015 n.s. |
| w=1 `expected_gain` | 0.6163 | +0.0190 n.s. | 0.5222 | +0.0035 n.s. |
| w=1 `one_minus_irrelevant` | 0.6049 | +0.0076 n.s. | 0.5148 | -0.0039 n.s. |

`***` p<0.001, `**` p<0.01, `*` p<0.05.

**Read both metric families.** `raw` scores unjudged documents as 0 (punishing a
system for surfacing trials TREC never pooled); `condensed` deletes them first
(flattering a system that surfaces many). Neither is honest alone — the gap between
them, read against `judged@10`, says how far apart they can be. Where the two agree
the result is robust; at w=1.0 they diverge sharply and `raw` loses significance
entirely.

## Local usage

```python
from structct import InferencePipeline
from structct.data import load_patient_graph, load_trial_graph

patient_graph = load_patient_graph(patient_id=1, year=2021)
inclusion_graph, exclusion_graph = load_trial_graph("NCT00000106")
pipeline = InferencePipeline("cv", fold=0)
result = pipeline.score(
    patient_graph,
    inclusion_graph,
    exclusion_graph,
    patient_text=patient_text,
    trial_text=trial_text,
)
print(result["class_probs"])
```

See the repository-level README for setup, CLI usage, and input requirements.

## Training

- **Data:** TREC Clinical Trials 2021–23 qrels/topics; eligibility criteria parsed into graphs (CHIA-style entities/relations)
- **Objective:** cross-entropy + **LambdaRank** (k=10) on topic-grouped batches (4 topics/batch)
- **Hard negatives:** 40/topic drawn from the retrieval candidate distribution (+5,307 samples), mined from *hybrid gemma-med* run files
- **Text branch:** the MedEmbed text encoder, whole-document embeddings of patient description and trial title+summary
- **Splits:** 5-fold CV over patients, seed fixed; each fold's test topics never seen in its training

## Limitations

- **The first stage is the ceiling.** medembed *with* reranking still scores below gemma-med *without* it. A better retriever buys more than more reranker tuning.
- **Do not use w=1.0.** Pure model ranking loses significance on both retrievers (p=0.65 / 0.85). It ranks better inside the judged pool but drifts out of it — `judged@10` falls from 0.92 to 0.80.
- **The hard-negative gain is retriever-sensitive.** +0.0175 (p=0.007) on gemma-med, +0.0148 (p=0.050) on medembed — the negatives were mined from gemma-med candidates and transfer only partly.
- **Within-collection evidence only.** 162 topics, one corpus, one set of assessors. Nothing about generalisation to another trial collection follows.
- **Not a clinical device.** Research artefact. It does not determine patient eligibility and must not be used to make enrolment decisions.

## Citation & provenance

Trained on TREC Clinical Trials track data (2021–23). Users must comply with the
TREC data usage agreements for the underlying collection and relevance judgments.

Each tensor-only checkpoint has an adjacent JSON sidecar containing its full
SHA-256 digest and the exact inference configuration.
