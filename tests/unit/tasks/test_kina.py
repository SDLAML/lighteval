from copy import deepcopy
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai.dataset import MemoryDataset

from lighteval.models.model_output import ModelResponse
from lighteval.tasks import lighteval_task
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.tasks import kina


CONFIGS = {config.name: config for config in kina.TASKS_TABLE}
VARIANTS = ("cf", "mcf", "mcf_em", "cot")
LABELS = "ABCDEFGHIJ"
DISCIPLINES = {
    "agronomy": "Agronomy",
    "economics": "Economics",
    "education": "Education",
    "engineering": "Engineering",
    "history": "History",
    "law": "Law",
    "literature_and_arts": "Literature and Arts",
    "management": "Management",
    "medicine": "Medicine",
    "philosophy": "Philosophy",
    "science": "Science",
    "sociology": "Sociology",
}
EXPECTED_NAMES = {f"kina:{discipline}:{variant}" for discipline in DISCIPLINES for variant in VARIANTS}


@pytest.fixture
def record():
    return {
        "index": 0,
        "discipline": "Engineering/Computer Science/Artificial Intelligence",
        "question": " Which option is correct? ",
        # Explicit keys, rather than the order of the list, determine labels.
        "options": [
            {"key": label, "answer": f" Option {label} ", "explanation": f"SECRET explanation {label}"}
            for label in reversed(LABELS)
        ],
        "correct_answer": "J",
    }


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("answer", list(LABELS))
def test_prompts_map_option_keys_to_gold_without_leaking_explanations(record, variant, answer):
    record["correct_answer"] = answer
    original = deepcopy(record)
    config = CONFIGS[f"kina:engineering:{variant}"]
    doc = config.prompt_function(record, task_name=config.name)

    assert record == original
    assert doc.task_name == config.name
    assert doc.gold_index == LABELS.index(answer)
    assert doc.specific == {"index": 0, "discipline": record["discipline"]}
    assert "SECRET" not in doc.query + "".join(doc.choices)
    if variant == "cf":
        assert doc.query == "Question: Which option is correct?\nAnswer:"
        assert doc.choices == [f" Option {label}" for label in LABELS]
        assert doc.get_golds() == [f" Option {answer}"]
    else:
        assert doc.choices == [f" {label}" for label in LABELS]
        assert doc.get_golds() == [f" {answer}"]
        for label in LABELS:
            assert f"\n{label}: Option {label}" in doc.query
    if variant == "cot":
        assert "Answer: $LETTER" in doc.instruction
        assert "Think step by step" in doc.instruction
        assert doc.query.startswith(doc.instruction)


@pytest.mark.parametrize("variant", ["mcf_em", "cot"])
@pytest.mark.parametrize("answer", list(LABELS))
def test_generative_scoring_supports_all_ten_labels(record, variant, answer):
    record["correct_answer"] = answer
    config = CONFIGS[f"kina:engineering:{variant}"]
    doc = config.prompt_function(record)
    wrong_answer = "A" if answer != "A" else "B"
    for prediction, expected in [(answer, 1.0), (wrong_answer, 0.0), ("", 0.0)]:
        text = f"Reasoning leads to this result.\nAnswer: {prediction}" if variant == "cot" else f" {prediction}"
        score = config.metrics[0].sample_level_fn.compute(doc, ModelResponse(text=[text]))
        assert score == expected


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("A is plausible.\nAnswer: J", 1.0),
        ("J is plausible.\nAnswer: A", 0.0),
        ("Answer: K", 0.0),
    ],
)
def test_cot_scoring_uses_the_final_valid_answer(record, prediction, expected):
    config = CONFIGS["kina:engineering:cot"]
    doc = config.prompt_function(record)
    assert config.metrics[0].sample_level_fn.compute(doc, ModelResponse(text=[prediction])) == expected


def test_inspect_sample_and_fewshot_exclude_explanations(record):
    sample = kina.record_to_sample(record)
    assert sample.id == 0
    assert sample.input == "Which option is correct?"
    assert sample.choices == [f"Option {label}" for label in LABELS]
    assert sample.target == "J"
    assert sample.metadata == {"discipline": record["discipline"]}
    fewshot = kina.sample_to_fewshot(sample)
    assert "A) Option A" in fewshot
    assert "J) Option J" in fewshot
    assert fewshot.endswith("ANSWER: J")
    assert "SECRET" not in fewshot


@pytest.mark.parametrize("discipline", DISCIPLINES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_loading_filters_evaluation_and_optional_fewshots(monkeypatch, record, discipline, variant):
    root = DISCIPLINES[discipline]
    rows = [
        {**record, "index": i, "question": f"Question {i}", "discipline": path}
        for i, path in enumerate(
            [
                *(f"{label}/Field/Subfield" for label in DISCIPLINES.values()),
                root,
                f"{root}/Another Field/Another Subfield",
                f"{root} Extra/Field",
                f"Other/{root}/Subfield",
            ]
        )
    ]
    dataset = DatasetDict({"test": Dataset.from_list(rows)})
    monkeypatch.setattr(lighteval_task, "load_dataset", lambda *args, **kwargs: dataset)
    config = CONFIGS[f"kina:{discipline}:{variant}"]
    task = LightevalTask(config)
    expected_paths = [f"{root}/Field/Subfield", root, f"{root}/Another Field/Another Subfield"]
    for docs in (task.eval_docs(), task.fewshot_docs()):
        assert [doc.specific["discipline"] for doc in docs] == expected_paths
        assert all(doc.get_golds() == ([" Option J"] if variant == "cf" else [" J"]) for doc in docs)
    samples = MemoryDataset([config.sample_fields(row) for row in rows]).filter(config.filter)
    assert [sample.metadata["discipline"] for sample in samples] == expected_paths


def test_discovery_and_task_selection(monkeypatch):
    configs = Registry._load_from_files([Path(kina.__file__)], "lighteval.tasks.tasks")
    assert len(kina.TASKS_TABLE) == len(EXPECTED_NAMES) == 48
    assert set(configs) == EXPECTED_NAMES
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: configs))
    selections = [
        ("kina", EXPECTED_NAMES),
        *[
            (f"kina:{variant}", {name for name in EXPECTED_NAMES if name.endswith(f":{variant}")})
            for variant in VARIANTS
        ],
        *[
            (f"kina:{discipline}", {f"kina:{discipline}:{variant}" for variant in VARIANTS})
            for discipline in DISCIPLINES
        ],
        *[(name, {name}) for name in sorted(EXPECTED_NAMES)],
    ]
    for selection, expected in selections:
        registry = Registry(f"{selection}|0")
        assert set(registry.load_tasks()) == {f"{name}|0" for name in expected}
    for config in configs.values():
        assert config.hf_repo == "2077AIDataFoundation/KINA"
        assert config.hf_subset == "default"
        assert config.hf_avail_splits == ("test",)
        assert config.evaluation_splits == ("test",)
