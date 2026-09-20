# Reranking candidate graphs

This dataset contains parsed inclusion and exclusion graphs for 13,229 trials in
the candidate pool retrieved for TREC Clinical Trials 2021–2023. It lets reviewers
run StructCT reranking without regenerating graphs with a large language model.

It is a candidate set rather than a complete trial corpus: trials never returned
by the two first-stage retrievers are absent. Every row has a non-empty inclusion
graph; 859 rows (6.5%) have an empty exclusion graph. See the full card at
[`data/reranking/README.md`](../../data/reranking/README.md) for the column schema,
coverage statistics, provenance, and limitations.
