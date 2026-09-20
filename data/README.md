# Dataset inventory

The three datasets are bundled as ordinary Parquet files and are read directly by
PyArrow. They require no dataset hub, loader script, network connection, or account.

| Directory | Rows | Purpose | Documentation |
|---|---:|---|---|
| `patient-graphs` | 165 | Parsed TREC CT patient descriptions, split by year | [Card](patient-graphs/README.md) |
| `trial-graphs` | 103,878 | Inclusion/exclusion graphs for the complete filtered trial set | [Card](trial-graphs/README.md) |
| `reranking` | 13,229 | Wide-format trial graphs for retrieved candidates | [Card](reranking/README.md) |

Entity and relation columns contain JSON strings because graph records have
heterogeneous optional fields. The Python package parses these fields into typed
Pydantic models.

```python
from structct.data import load_patient_graph, load_trial_graph

patient = load_patient_graph(patient_id=1, year=2021)
inclusion, exclusion = load_trial_graph("NCT00000106")
```
