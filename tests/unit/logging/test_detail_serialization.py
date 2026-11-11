import json
from dataclasses import asdict, replace

import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.dummy.dummy_model import DummyModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.multilingual.tasks import simpleqa_verified
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.kina import TASKS_TABLE


@pytest.fixture
def tracker(tmp_path):
    tracker = EvaluationTracker(output_dir=str(tmp_path), save_details=True)
    tracker.general_config_logger.log_model_info(DummyModelConfig(model_name="test-model"))
    return tracker


@pytest.mark.parametrize(
    "config",
    [config for config in simpleqa_verified.TASKS_TABLE if config.name.endswith(":gen")],
    ids=lambda config: config.name,
)
def test_results_save_with_partial_normalizers(tracker, tmp_path, config):
    tracker.task_config_logger.tasks_configs = {config.name: config}
    tracker.metrics_logger.metric_aggregated = {config.name: {metric.metric_name: 1.0 for metric in config.metrics}}

    tracker.save()

    result_file = next((tmp_path / "results").rglob("results_*.json"))
    saved_results = json.loads(result_file.read_text())
    assert saved_results["results"] == tracker.metrics_logger.metric_aggregated
    saved_metrics = saved_results["config_tasks"][config.name]["metrics"]
    assert len(saved_metrics) == len(config.metrics)
    for metric in saved_metrics:
        assert "_normalize_answer" in metric["sample_level_fn"]


@pytest.mark.parametrize("num_fewshots", [0, 1])
@pytest.mark.parametrize("save_details", [False, True])
def test_kina_results_and_details_round_trip(tracker, tmp_path, num_fewshots, save_details):
    config = next(config for config in TASKS_TABLE if config.name == "kina:engineering:cot")
    task = LightevalTask(replace(config, num_fewshots=num_fewshots))
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_list(
                [
                    {
                        "index": index,
                        "discipline": "Engineering/Computer Science/Artificial Intelligence",
                        "question": f"Question {index}?",
                        "options": [{"key": label, "answer": f"Option {label}"} for label in "ABCDEFGHIJ"],
                        "correct_answer": "J",
                    }
                    for index in range(2)
                ]
            )
        }
    )
    docs = task.get_docs()
    for doc in docs:
        response = ModelResponse(input=doc.query, input_tokens=[1, 2], text=["Answer: J"], output_tokens=[[3, 4]])
        score = config.metrics[0].sample_level_fn.compute(doc, response)
        tracker.details_logger.log(task.full_name, doc, response, {"extractive_match": score})
    original_docs = [asdict(doc) for doc in docs]
    tracker.metrics_logger.metric_aggregated = {task.full_name: {"extractive_match": 1.0}}
    tracker.should_save_details = save_details

    tracker.save()

    assert [asdict(doc) for doc in docs] == original_docs
    result_files = list((tmp_path / "results").rglob("results_*.json"))
    assert len(result_files) == 1
    assert json.loads(result_files[0].read_text())["results"] == tracker.metrics_logger.metric_aggregated
    detail_files = list((tmp_path / "details").rglob("*.parquet"))
    if not save_details:
        assert detail_files == []
        return
    assert len(detail_files) == 1
    saved_details = pq.read_table(detail_files[0]).to_pylist()
    assert len(saved_details) == 2
    for saved, doc in zip(saved_details, docs):
        assert saved["doc"]["source_id"] == {"split": "test", "row_index": doc.source_id[1]}
        assert saved["doc"]["specific"] == doc.specific
        assert saved["model_response"]["input_tokens"] == [1, 2]
        assert saved["model_response"]["text"] == ["Answer: J"]
        assert saved["metric"] == {"extractive_match": 1.0}
        assert len(saved["doc"]["fewshot_samples"]) == num_fewshots
        for sample in saved["doc"]["fewshot_samples"]:
            assert sample["source_id"]["split"] == "test"
            assert sample["source_id"]["row_index"] != doc.source_id[1]
            assert isinstance(sample["source_id"]["row_index"], int)


def test_saved_details_allow_missing_and_nested_source_identities(tracker, tmp_path):
    nested = Doc("Nested example?", ["Yes"], 0, source_id=("train", 4))
    fewshot = Doc("Example?", ["Yes"], 0, source_id=("dev", 3), fewshot_samples=[nested])
    doc = Doc("Question?", ["Yes"], 0, fewshot_samples=[fewshot], specific={"source_id": "original-reference"})
    tracker.details_logger.log("manual|1", doc, ModelResponse(text=["Yes"]), {"accuracy": 1.0})

    tracker.save()

    detail_file = next((tmp_path / "details").rglob("*.parquet"))
    saved_doc = pq.read_table(detail_file).to_pylist()[0]["doc"]
    assert saved_doc["source_id"] is None
    assert saved_doc["specific"] == {"source_id": "original-reference"}
    saved_fewshot = saved_doc["fewshot_samples"][0]
    assert saved_fewshot["source_id"] == {"split": "dev", "row_index": 3}
    assert saved_fewshot["fewshot_samples"][0]["source_id"] == {"split": "train", "row_index": 4}
    assert (doc.source_id, fewshot.source_id, nested.source_id) == (None, ("dev", 3), ("train", 4))
