from copy import deepcopy
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai.dataset import MemoryDataset

from lighteval.models.model_output import ModelResponse
from lighteval.tasks import lighteval_task
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.tasks import rbench


CONFIGS = {config.name: config for config in rbench.TASKS_TABLE}


@pytest.fixture
def record():
    return {
        "index": "1",
        "question": " Which option is correct? ",
        "answer": "F",
        **{label: f"Option {label}" for label in "ABCDEF"},
    }


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("variant", ["cf", "mcf", "mcf_em", "cot"])
def test_prompts_preserve_options_and_answer_mapping(record, language, variant):
    original = deepcopy(record)
    config = CONFIGS[f"rbench:{language}:{variant}"]
    doc = config.prompt_function(record, task_name=config.name)
    assert doc.task_name == config.name
    assert doc.gold_index == 5
    assert doc.images is None
    if variant == "cf":
        prefix, suffix = ("Question:", "Answer:") if language == "en" else ("问题：", "答案：")
        assert doc.query == f"{prefix} Which option is correct?\n{suffix}"
        assert doc.choices == [f" Option {label}" for label in "ABCDEF"]
        assert doc.get_golds() == [" Option F"]
    else:
        assert doc.choices == [f" {label}" for label in "ABCDEF"]
        assert doc.get_golds() == [" F"]
        for label in "ABCDEF":
            assert f"{label}: Option {label}" in doc.query
    if variant == "cot":
        assert "Answer: $LETTER" in doc.instruction
        assert doc.query.startswith(doc.instruction)
    assert record == original


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("variant", ["mcf_em", "cot"])
@pytest.mark.parametrize("answer", list("ABCDEF"))
def test_generative_metrics_score_each_label(record, language, variant, answer):
    record["answer"] = answer
    config = CONFIGS[f"rbench:{language}:{variant}"]
    doc = config.prompt_function(record)
    for prediction, expected in [(answer, 1.0), ("A" if answer != "A" else "B", 0.0)]:
        text = f"Reasoning leads to this result.\nAnswer: {prediction}" if variant == "cot" else f" {prediction}"
        score = config.metrics[0].sample_level_fn.compute(doc, ModelResponse(text=[text]))
        assert score == expected


def test_inspect_samples_and_fewshots(record):
    sample = rbench.record_to_sample(record)
    assert sample.id == "1"
    assert sample.input == "Which option is correct?"
    assert sample.choices == [f"Option {label}" for label in "ABCDEF"]
    assert sample.target == "F"
    fewshot = rbench.sample_to_fewshot(sample)
    assert "F) Option F" in fewshot
    assert fewshot.endswith("ANSWER: F")


@pytest.mark.parametrize(
    "answer, expected", [("A", True), (" F\n", True), ("E\n\nF", False), ("", False), ("G", False)]
)
def test_ambiguous_answers_are_filtered_for_both_backends(record, answer, expected):
    record["answer"] = answer
    assert rbench.rbench_filter(record) is expected
    sample = rbench.record_to_sample(record)
    for config in CONFIGS.values():
        assert config.filter(sample) is expected


@pytest.mark.parametrize("language", ["en", "zh"])
def test_dataset_loading_filters_evaluation_and_fewshots(monkeypatch, record, language):
    rows = [record, {**record, "index": "236", "answer": "E\n\nF"}]
    dataset = DatasetDict({"test": Dataset.from_list(rows)})
    monkeypatch.setattr(lighteval_task, "load_dataset", lambda *args, **kwargs: dataset)
    config = CONFIGS[f"rbench:{language}:mcf"]
    task = LightevalTask(config)
    for docs in (task.eval_docs(), task.fewshot_docs()):
        assert len(docs) == 1
        assert docs[0].get_golds() == [" F"]
    samples = MemoryDataset([config.sample_fields(row) for row in rows]).filter(config.filter)
    assert [sample.target for sample in samples] == ["F"]


def test_module_discovery_and_text_subset_selection(monkeypatch):
    configs = Registry._load_from_files([Path(rbench.__file__)], "lighteval.tasks.tasks")
    expected_names = {
        f"rbench:{language}:{variant}" for language in ("en", "zh") for variant in ("cf", "mcf", "mcf_em", "cot")
    }
    assert set(configs) == expected_names
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: configs))
    for selection, expected in [
        ("rbench", expected_names),
        ("rbench:en", {name for name in expected_names if ":en:" in name}),
        ("rbench:mcf", {"rbench:en:mcf", "rbench:zh:mcf"}),
        ("rbench:zh:cot", {"rbench:zh:cot"}),
    ]:
        registry = Registry(f"{selection}|0")
        assert set(registry.load_tasks()) == {f"{name}|0" for name in expected}
    for config in configs.values():
        language = config.name.split(":")[1]
        assert config.hf_repo == "R-Bench/R-Bench"
        assert config.hf_subset == f"rbench-t_{language}"
        assert config.hf_avail_splits == ("test",)
        assert config.evaluation_splits == ("test",)
