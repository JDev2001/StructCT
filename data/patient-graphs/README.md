# Clinical Trials Patient Graphs

Graph representations of the 165 patient descriptions used in the TREC Clinical
Trials 2021–2023 editions. The data is split into one Parquet file per year.

| Split | Examples | Uncompressed bytes |
|---|---:|---:|
| 2021 | 75 | 233,448 |
| 2022 | 50 | 82,015 |
| 2023 | 40 | 71,432 |

## Schema

| Column | Type | Description |
|---|---|---|
| `patient_id` | int64 | TREC topic identifier within the edition |
| `year` | int64 | TREC Clinical Trials edition |
| `entities` | string (JSON) | Extracted entities from the patient description |
| `relations` | string (JSON) | Typed relations between entity identifiers |

The graph records are model-generated representations. They may contain extraction
errors and must not be treated as curated clinical facts or used for patient care.

See [the extended dataset notes](../../docs/datasets/patient-graphs.md).
