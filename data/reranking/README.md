---
license: cc-by-nc-4.0
language: [en]
tags:
  - clinical-trials
  - information-extraction
  - knowledge-graph
  - biomedical
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files: data/graphs.parquet
---

# Clinical Trial Eligibility Graphs (rerank candidate set)

Parsed eligibility criteria for **13,229 ClinicalTrials.gov trials** — the complete
candidate set retrieved across TREC Clinical Trials 2021–23 by two first-stage
retrievers. Each trial's inclusion and exclusion criteria are represented as
entities and typed relations.

This exists so the
[bundled CrossGAT eligibility model](../../models/crossgat-trial-eligibility-cv/README.md)
can be run **without an LLM**. Generating these graphs took ~28 GPU-hours on a
20B-parameter model; this dataset is that output, so reproduction costs nothing.

## Schema

| column | type | description |
|---|---|---|
| `nct_id` | string | ClinicalTrials.gov identifier |
| `inclusion_entities` | string (JSON) | entities parsed from inclusion criteria |
| `inclusion_relations` | string (JSON) | typed relations between them |
| `exclusion_entities` | string (JSON) | entities parsed from exclusion criteria |
| `exclusion_relations` | string (JSON) | typed relations between them |
| `n_inc_entities`, `n_inc_relations` | int | counts, for filtering without parsing |
| `n_exc_entities`, `n_exc_relations` | int | counts |

Graph columns are JSON strings rather than nested structs: entity and relation
schemas are heterogeneous (optional `code_expr`, variable relation arity), and
flattening them into a fixed Arrow schema would lose fields. Parse on read.

```python
import json
import pyarrow.parquet as pq

table = pq.read_table("data/reranking/data/graphs.parquet")
row = table.slice(0, 1).to_pylist()[0]
inc_entities = json.loads(row["inclusion_entities"])
inc_relations = json.loads(row["inclusion_relations"])
```

## Coverage and known gaps

- **13,229 trials**, all with a non-empty inclusion graph.
- **859 trials (6.5%) have an empty exclusion graph.** Most genuinely state no
  exclusion criteria; some are parse failures. The model handles an empty side —
  it is trained on data containing them — but filter on `n_exc_entities` if you
  need both sides populated.
- Median 25 inclusion entities and 23 exclusion entities per trial.
- **This is a candidate set, not a corpus.** It covers what two specific retrievers
  surfaced in the top-100 for TREC CT topics, not ClinicalTrials.gov at large. A
  trial neither retriever ever retrieved is absent.

## Provenance

Generated with `openai/gpt-oss-20b` via a CHIA-style extraction prompt, over the
ClinicalTrials.gov snapshot used by TREC Clinical Trials 2021–23. Cached by NCT id
during benchmark runs and consolidated here; 182 MB of JSON, 17.7 MB as
zstd-compressed Parquet. A random 200-trial sample was verified to round-trip
byte-identically.

One trial in the original candidate set, `NCT03595566`, produced no graph on any
attempt and is absent.

## Intended use

Reranking and eligibility-classification research on TREC CT. Users must comply
with the TREC data usage agreements for the underlying collection. Extracted
criteria are model output, not curated annotations — they contain extraction
errors and must not be used for clinical decisions.
