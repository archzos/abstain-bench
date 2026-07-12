from abstain_bench.datasets.loaders import load_dataset


def test_multilingual_pack_exposes_language_metadata() -> None:
    rows = load_dataset("multilingual-pack", limit=10)
    languages = {row.metadata.get("language") for row in rows}
    assert {"en", "es", "fr", "hi", "pt"}.issubset(languages)
