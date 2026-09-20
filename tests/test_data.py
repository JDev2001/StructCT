from structct.data.local import load_patient_graph, summarize_datasets


def test_dataset_inventory() -> None:
    summaries = {item["name"]: item for item in summarize_datasets()}
    assert summaries["patient-graphs"]["rows"] == 165
    assert summaries["trial-graphs"]["rows"] == 103_878
    assert summaries["reranking"]["rows"] == 13_229


def test_patient_graph_can_be_loaded() -> None:
    graph = load_patient_graph(patient_id=1, year=2021)
    assert graph.entities
