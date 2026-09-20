import pytest

from structct.inference.loader import load_model


@pytest.mark.parametrize(
    ("model", "classes"),
    [
        ("2class-base", ("excluded", "eligible")),
        ("cv", ("irrelevant", "excluded", "eligible")),
    ],
)
def test_sanitized_checkpoint_loads(model: str, classes: tuple[str, ...]) -> None:
    loaded = load_model(model, fold=0)
    assert loaded.classes == classes
    assert loaded.checkpoint_path.name == "checkpoint.pt"
