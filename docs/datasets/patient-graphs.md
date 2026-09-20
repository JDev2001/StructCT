# Patient graphs

This dataset contains graph representations of the 165 patient descriptions used
in the TREC Clinical Trials 2021, 2022, and 2023 editions.

| Split/year | Rows |
|---|---:|
| 2021 | 75 |
| 2022 | 50 |
| 2023 | 40 |

## Schema

| Column | Type | Description |
|---|---|---|
| `patient_id` | int64 | Topic identifier within the TREC edition |
| `year` | int64 | TREC Clinical Trials edition |
| `entities` | string (JSON) | Extracted patient entities |
| `relations` | string (JSON) | Typed relations between entity identifiers |

Patient graphs are model-generated structured representations, not clinical
records or curated medical annotations. They must not be used for care decisions.
