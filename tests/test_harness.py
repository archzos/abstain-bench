import json
from pathlib import Path

from abstain_bench.adapters.base import ModelAdapter
from abstain_bench.harness import run_benchmark
from abstain_bench.models import GenerationResult


class FakeAdapter(ModelAdapter):
    def generate(self, prompt: str) -> GenerationResult:
        if "capital of france" in prompt.lower():
            return GenerationResult(model_name=self.name, text="Paris", confidence=95.0)
        return GenerationResult(model_name=self.name, text="I don't know", confidence=15.0)


def test_harness_dry_run(tmp_path: Path) -> None:
    config = {
        "models": [{"name": "fake-model", "provider": "vllm", "model_id": "fake-id"}],
        "weights": {
            "lambda_confident_wrong": 2.0,
            "lambda_unwarranted_abstain": 0.5,
            "lambda_correct_abstain": 1.0,
        },
        "abstention": {"confidence_threshold": 40.0, "hedge_patterns": ["i don't know"]},
        "harness": {"max_questions": 5, "lm_eval_baseline": {"enabled": False}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_benchmark(
        config_path=str(config_path),
        output_db=str(tmp_path / "results.duckdb"),
        dataset_name="simpleqa-subset",
        model_names=["fake-model"],
        dry_run=True,
    )

    assert result.run_id == "dry-run"
    assert result.total_rows == 0


def test_harness_end_to_end_with_mocked_adapter(tmp_path: Path, monkeypatch) -> None:
    config = {
        "models": [{"name": "fake-model", "provider": "vllm", "model_id": "fake-id"}],
        "weights": {
            "lambda_confident_wrong": 2.0,
            "lambda_unwarranted_abstain": 0.5,
            "lambda_correct_abstain": 1.0,
        },
        "abstention": {
            "confidence_threshold": 40.0,
            "hedge_patterns": ["i don't know", "cannot determine"],
        },
        "harness": {"max_questions": 6, "lm_eval_baseline": {"enabled": False}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr("abstain_bench.harness._build_adapter", lambda _cfg: FakeAdapter("fake-model", "fake-id"))

    output_db = tmp_path / "results.duckdb"
    result = run_benchmark(
        config_path=str(config_path),
        output_db=str(output_db),
        dataset_name="simpleqa-with-unanswerable",
        model_names=["fake-model"],
        dry_run=False,
    )

    assert result.total_rows > 0
    assert len(result.summaries) == 1
    assert output_db.exists()
