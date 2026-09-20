# Model inventory

All StructCT checkpoints are stored locally as tensor-only PyTorch state
dictionaries. No checkpoint is downloaded during inference.

| Directory | Alias | Description |
|---|---|---|
| `crossgat-trial-eligibility-full` | `full` | Full-data 3-class deployment refit |
| `crossgat-trial-eligibility-cv` | `cv` | 5-fold 3-class model with LambdaRank and hard negatives |
| `crossgat-trial-eligibility-3class-ce` | `3class-ce` | 5-fold 3-class cross-entropy control |
| `crossgat-trial-eligibility-2class-base` | `2class-base` | 5-fold 2-class cross-entropy control |
| `crossgat-trial-eligibility-2class-eligloss` | `2class-eligloss` | 5-fold 2-class eligibility-loss ablation |
| `crossgat-trial-eligibility-2class-rank` | `2class-rank` | 5-fold 2-class ranking-loss ablation |

Each cross-validation directory contains `fold0` through `fold4`, each with one
`checkpoint.pt`. Read each directory's model card for training details, reported
results, intended use, and limitations.
