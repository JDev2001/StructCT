# StructCT

StructCT is a self-contained research artifact for structured clinical-trial
eligibility matching. It bundles the released graph datasets, all trained
StructCT/CrossGAT checkpoints, and inference code in one repository. Reviewers do
not need access to the original model or dataset repositories.

The system represents patient descriptions and trial inclusion/exclusion criteria
as typed graphs, encodes them with a graph-attention network, and predicts
eligibility labels. Its intended role is reranking candidates returned by a
first-stage retriever; it is not a standalone retriever.

> **Research use only.** StructCT is not a clinical device and must not be used to
> determine eligibility, recommend enrollment, or make medical decisions.

## What is included

| Component | Contents |
|---|---|
| Trial graphs | 103,878 inclusion/exclusion graph rows stored in two Parquet shards |
| Reranking graphs | 13,229 trial graphs from the TREC CT candidate pool |
| Patient graphs | 165 patient graphs across the 2021–2023 TREC CT editions |
| Cross-validation models | Five fold checkpoints |
| Ablation models | 2-class cross-entropy, eligibility-loss, ranking-loss, and 3-class cross-entropy variants |
| Deployment model | One 3-class checkpoint refit on all labeled pairs |
| Inference package | Local data readers, graph preprocessing, model loading, scoring, and a CLI |

The only model fetched at runtime is the public third-party
[`abhinand/MedEmbed-base-v0.1`](https://huggingface.co/abhinand/MedEmbed-base-v0.1)
text encoder. None of the StructCT checkpoints or datasets is downloaded at
runtime. After MedEmbed is cached once, inference can run offline.

## Quick start with uv

Install [uv](https://docs.astral.sh/uv/), clone this repository, and run:

```bash
uv sync --extra dev
uv run structct inventory
uv run structct data-summary
uv run pytest
```

The lockfile fixes the complete Python dependency graph. CPU PyTorch wheels are
used by default to keep the review setup portable.

## Score a bundled graph pair

The 2-class checkpoints are graph-only and can be exercised directly with the
bundled Parquet data:

```bash
uv run structct score \
  --model 2class-base \
  --fold 0 \
  --year 2021 \
  --patient-id 1 \
  --trial-id NCT00000106
```

Equivalent Python:

```python
from structct import InferencePipeline
from structct.data import load_patient_graph, load_trial_graph

patient = load_patient_graph(patient_id=1, year=2021)
inclusion, exclusion = load_trial_graph("NCT00000106")

pipeline = InferencePipeline("2class-base", fold=0)
result = pipeline.score(patient, inclusion, exclusion)
print(result["class_probs"])
```

The `full`, `cv`, and `3class-ce` models also use whole-document text embeddings.
Pass the authorized TREC patient description and trial title/summary with
`--patient-text-file` and `--trial-text-file`. These source texts are not
redistributed because they are governed by the TREC collection terms.

## Model selection

| Alias | Classes | Checkpoints | Training objective | Recommended use |
|---|---:|---:|---|---|
| `full` | 3 | 1 | Cross-entropy + LambdaRank + hard negatives | Deployment after cross-validation |
| `cv` | 3 | 5 | Cross-entropy + LambdaRank + hard negatives | Reproduce reported 5-fold results |
| `3class-ce` | 3 | 5 | Focal cross-entropy | Three-class control |
| `2class-base` | 2 | 5 | Focal cross-entropy | Two-class control |
| `2class-eligloss` | 2 | 5 | Cross-entropy + eligibility margin | Auxiliary-loss ablation |
| `2class-rank` | 2 | 5 | LambdaRank + retrieval negatives | Ranking-loss ablation |

## Repository layout

```text
StructCT/
├── data/                     # Bundled Parquet datasets and dataset cards
│   ├── patient-graphs/
│   ├── reranking/
│   └── trial-graphs/
├── models/                   # Sanitized inference weights and model cards
├── src/structct/
│   ├── data/                 # Schemas, readers, and graph construction
│   ├── inference/            # Checkpoint loader and scoring pipeline
│   └── model/                # CrossGAT architecture
├── docs/                     # Dataset and reproducibility documentation
├── tests/                    # Fast artifact integrity tests
├── artifact-manifest.json    # Inventory, file sizes, and SHA-256 hashes
├── pyproject.toml
└── uv.lock
```

## Reproducibility and privacy

The published `.pt` files contain model tensors only. Optimizer states, training
logs, machine paths, usernames, cache locations, and experiment-tracking metadata
were removed from the original Lightning checkpoints. Every weight file has an
adjacent JSON sidecar containing only the inference configuration, class order,
architecture type, and SHA-256 digest.

No token, credential, email address, personal path, account handle, or link to the
original StructCT hosting repositories is included. See
[Reproducibility notes](docs/reproducibility.md) for the audit procedure.

## Data and licensing

The model weights and reranking graph release are marked CC BY-NC 4.0. The TREC
Clinical Trials source collection and judgments remain subject to their own usage
terms. Dataset cards without an explicit upstream license are not relicensed here.
See [LICENSE.md](LICENSE.md) and [dataset documentation](data/README.md) before use.
