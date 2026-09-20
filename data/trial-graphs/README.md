# Clinical Trial Eligibility Graphs

Graph representations of clinical-trial inclusion and exclusion criteria. The
dataset contains 103,878 rows in two compressed Parquet shards (545,681,858 bytes
after decompression).

## Schema

| Column | Type | Description |
|---|---|---|
| `nct_id` | string | ClinicalTrials.gov trial identifier |
| `type` | string | `inclusion` or `exclusion` |
| `entities` | string (JSON) | Extracted eligibility entities |
| `relations` | string (JSON) | Typed relations between entity identifiers |

JSON strings are used instead of fixed nested Arrow structs because entity records
contain optional numeric expressions and relation records vary in structure.

These are automatically generated graphs and may contain extraction errors. See
[the extended dataset notes](../../docs/datasets/trial-graphs.md).
