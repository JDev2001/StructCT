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

# CrossGAT Trial Eligibility — full-data model

The deployment checkpoint: a 3-class eligibility model that **reranks clinical-trial
retrieval candidates**, refit on **every labelled pair** after cross-validation.

This is the standard final step after CV — cross-validation validated the procedure
and its hyperparameters, and the shipped model applies them to all the data. If you
want to *reproduce reported numbers*, or to ensemble, use the
[5-fold CV checkpoints](../crossgat-trial-eligibility-cv/README.md)
instead.

## ⚠️ No performance number can be reported for this checkpoint

It has seen every labelled pair in the collection, so it has **no held-out data left**
and no unbiased performance estimate. Any number you compute on TREC CT 2021–23 with
this model is contaminated.

**Cite the cross-validated estimate instead**, from the CV repo: graded NDCG@10
0.5973 → **0.6515** (+0.0542, p<0.0001, 162 topics) on hybrid BM25+gemma-med at the
recommended operating point. That is the expected performance of *this procedure*;
this checkpoint is the procedure applied to more data.

## Local inputs

This checkpoint consumes parsed patient and trial eligibility graphs. All StructCT checkpoints, graph data, and inference code are bundled locally; none is fetched from the original model or dataset repositories. It also requires the original patient description and trial title/summary. Those TREC texts are not redistributed and must be supplied by an authorized reviewer.

## Recommended configuration

```
score = (1 - w) * retrieval_norm + w * p_eligible_norm      with w = 0.5
```

Both terms min-max normalised within a topic. Never replace the first stage
outright (w = 1.0): that loses significance on both retrievers tested. The model
reranks; it cannot retrieve — per-topic AUC for relevant-vs-rest is 0.533, chance.

## How the stopping point was chosen

With no validation split there is nothing to select on, so the epoch count was
**fixed in advance** from the CV runs. Averaging the per-fold validation *curves*
(not their argmaxes, which are noise around a plateau) puts `val_ndcg10`'s optimum
at epoch 3:

| epoch | 0 | 1 | 2 | **3** | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| mean val_ndcg10 | .602 | .623 | .615 | **.624** | .623 | .616 | .611 | .600 | .607 |

The curve is flat from epoch 1 to 5, so the exact choice inside that band is not
critical — which is the point. The median of the per-fold argmaxes would have said
epoch 2, but that averages noise rather than signal.

Refit epochs carry ~26% more gradient steps than a CV fold (1,633 vs ~1,294
batches), so epoch 3 here is ≈3.8 CV-equivalent epochs — still inside the plateau.

## Local usage

```python
from structct import InferencePipeline
from structct.data import load_patient_graph, load_trial_graph

patient_graph = load_patient_graph(patient_id=1, year=2021)
inclusion_graph, exclusion_graph = load_trial_graph("NCT00000106")
pipeline = InferencePipeline("full")
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

- **Data:** all 78,370 labelled pairs (73,063 from TREC CT 2021–23 qrels + 5,307 hard negatives)
- **Objective:** cross-entropy + **LambdaRank** (k=10) on topic-grouped batches (4 topics/batch)
- **Hard negatives:** 40/topic from the retrieval candidate distribution, mined from hybrid gemma-med run files
- **Text branch:** the MedEmbed text encoder
- **Epochs:** 4 (0–3), fixed in advance; no early stopping, no checkpoint selection

## Limitations

- **No unbiased performance estimate** — see the warning above.
- **The first stage is the ceiling.** A better retriever buys more than more reranker tuning.
- **The hard-negative gain is retriever-sensitive** (+0.0175 on gemma-med, +0.0148 on medembed): the negatives were mined from gemma-med candidates and transfer only partly.
- **Within-collection evidence only.** One corpus, one set of assessors.
- **Not a clinical device.** Research artefact. It does not determine patient eligibility and must not be used to make enrolment decisions.

## Citation & provenance

Trained on TREC Clinical Trials track data (2021–23). Users must comply with the
TREC data usage agreements for the underlying collection and relevance judgments.

Each sanitized checkpoint has a full SHA-256 digest in its adjacent JSON sidecar.
