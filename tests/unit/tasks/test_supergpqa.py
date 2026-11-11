from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai.dataset import MemoryDataset

from lighteval.models.model_output import ModelResponse
from lighteval.tasks import lighteval_task
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.tasks import mmlu, supergpqa
from lighteval.tasks.tasks.supergpqa import (
    TASKS_TABLE,
    record_to_sample,
    sample_to_fewshot,
    supergpqa_cf_prompt,
    supergpqa_cot,
    supergpqa_cot_prompt,
    supergpqa_mcf_em,
    supergpqa_mcf_prompt,
)


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
    "military_science": "Military Science",
    "philosophy": "Philosophy",
    "science": "Science",
    "sociology": "Sociology",
}
DIFFICULTIES = ("easy", "middle", "hard")
VARIANTS = ("cf", "mcf", "mcf_em", "cot")
CONFIGS = {config.name: config for config in TASKS_TABLE}
EXPECTED_NAMES = {
    f"supergpqa:{discipline}{suffix}:{variant}"
    for discipline in DISCIPLINES
    for suffix in ("", ":easy", ":middle", ":hard")
    for variant in VARIANTS
}


@pytest.fixture(params=[4, 6, 10])
def record(request):
    return {
        "question": " Which option is correct? ",
        "discipline": "Engineering",
        "difficulty": "hard",
        "options": [f"Option {i}" for i in range(request.param)],
        # The dataset's answer text can differ in capitalization from its option.
        "answer": f"option {request.param - 1}",
        "answer_letter": {4: "D", 6: "F", 10: "J"}[request.param],
    }


def test_prompts_preserve_options_and_use_answer_letter(record):
    original = deepcopy(record)
    cf = supergpqa_cf_prompt(record, task_name="supergpqa:cf")
    mcf = supergpqa_mcf_prompt(record, task_name="supergpqa:mcf")
    cot = supergpqa_cot_prompt(record, task_name="supergpqa:cot")
    expected_labels = list("ABCDEFGHIJ"[: len(record["options"])])

    assert cf.query == "Question: Which option is correct?\nAnswer:"
    assert cf.choices == [f" {option}" for option in record["options"]]
    assert cf.get_golds() == [f" {record['options'][-1]}"]
    assert mcf.choices == [f" {label}" for label in expected_labels]
    assert mcf.get_golds() == [f" {record['answer_letter']}"]
    assert cot.choices == [f" {label}" for label in expected_labels]
    assert cot.get_golds() == [f" {record['answer_letter']}"]
    assert f"LETTER is one of {''.join(expected_labels)}." in cot.instruction
    for label, option in zip(expected_labels, record["options"]):
        assert f"\n{label}: {option}" in mcf.query
        assert f"\n{label}: {option}" in cot.query
    for doc, variant in [(cf, "cf"), (mcf, "mcf"), (cot, "cot")]:
        assert doc.gold_index == len(record["options"]) - 1
        assert doc.task_name == f"supergpqa:{variant}"
    assert record == original


@pytest.mark.parametrize("prompt", [supergpqa_cf_prompt, supergpqa_mcf_prompt, supergpqa_cot_prompt])
def test_first_option_can_be_correct(record, prompt):
    record["answer_letter"] = "A"
    assert prompt(record).gold_index == 0


def test_inspect_sample_and_fewshot_use_letter_target(record):
    sample = record_to_sample(record)
    assert sample.input == record["question"].strip()
    assert sample.choices == record["options"]
    assert sample.target == record["answer_letter"]
    assert sample.metadata == {"discipline": "Engineering", "difficulty": "hard"}
    fewshot = sample_to_fewshot(sample)
    assert f"{sample.target}) {record['options'][-1]}" in fewshot
    assert fewshot.endswith(f"ANSWER: {sample.target}")


@pytest.mark.parametrize("correct", [True, False])
def test_reasoning_metric_scores_all_supported_answer_letters(record, correct):
    doc = supergpqa_cot_prompt(record)
    answer = record["answer_letter"] if correct else "A"
    response = ModelResponse(text=[f"After considering the options, I can conclude.\nAnswer: {answer}"])
    score = supergpqa_cot.metrics[0].sample_level_fn.compute(doc, response)
    assert score == float(correct)


@pytest.mark.parametrize("correct", [True, False])
def test_exact_match_metric_scores_answer_letters(record, correct):
    doc = supergpqa_mcf_prompt(record)
    answer = record["answer_letter"] if correct else "A"
    response = ModelResponse(text=[f" {answer}"])
    score = supergpqa_mcf_em.metrics[0].sample_level_fn.compute(doc, response)
    assert score == float(correct)


def test_supergpqa_module_discovery_and_selection(monkeypatch):
    expected_names = EXPECTED_NAMES
    assert len(TASKS_TABLE) == len(expected_names) == 208
    configs = Registry._load_from_files([Path(supergpqa.__file__)], "lighteval.tasks.tasks")
    assert set(configs) == expected_names
    # Keep this unit test independent of unrelated tasks and optional judge backends.
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: configs))
    registry = Registry("supergpqa|0")
    assert set(registry.task_to_configs) == expected_names
    assert set(registry.load_tasks()) == {f"{name}|0" for name in expected_names}
    for config in TASKS_TABLE:
        assert config.hf_repo == "m-a-p/SuperGPQA"
        assert config.hf_subset == "default"
        assert config.hf_avail_splits == ("train",)
        assert config.evaluation_splits == ("train",)


@pytest.mark.parametrize(
    ("selection", "expected_names"),
    [
        ("supergpqa:cf", {name for name in EXPECTED_NAMES if name.endswith(":cf")}),
        ("supergpqa:engineering", {name for name in EXPECTED_NAMES if name.startswith("supergpqa:engineering:")}),
        ("supergpqa:engineering:hard", {f"supergpqa:engineering:hard:{variant}" for variant in VARIANTS}),
        ("supergpqa:engineering:cf", {"supergpqa:engineering:cf"}),
        ("supergpqa:military_science:middle:cot", {"supergpqa:military_science:middle:cot"}),
    ],
)
def test_select_subtask_groups(monkeypatch, selection, expected_names):
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: CONFIGS))
    registry = Registry(f"{selection}|2")
    assert set(registry.task_to_configs) == expected_names
    assert set(registry.load_tasks()) == {f"{name}|2" for name in expected_names}


@pytest.mark.parametrize("discipline", DISCIPLINES)
@pytest.mark.parametrize("difficulty", [None, *DIFFICULTIES])
def test_subtasks_filter_rows_and_inspect_samples(discipline, difficulty):
    rows = [
        {
            "question": f"{label} {level}",
            "options": ["Yes", "No"],
            "answer_letter": "A",
            "discipline": label,
            "difficulty": level,
        }
        for label in DISCIPLINES.values()
        for level in DIFFICULTIES
    ]
    expected = [
        f"{DISCIPLINES[discipline]} {level}" for level in DIFFICULTIES if difficulty is None or level == difficulty
    ]
    for variant in VARIANTS:
        suffix = f":{difficulty}" if difficulty else ""
        config = CONFIGS[f"supergpqa:{discipline}{suffix}:{variant}"]
        assert [row["question"] for row in rows if config.hf_filter(row)] == expected
        samples = MemoryDataset([config.sample_fields(row) for row in rows]).filter(config.filter)
        assert [sample.input for sample in samples] == expected


def test_lighteval_filters_evaluation_and_fewshot_rows(monkeypatch):
    rows = [
        {
            "question": question,
            "options": ["Yes", "No"],
            "answer_letter": "A",
            "discipline": discipline,
            "difficulty": difficulty,
        }
        for question, discipline, difficulty in [
            ("Wrong discipline", "Science", "hard"),
            ("Wrong difficulty", "Engineering", "easy"),
            ("Selected question", "Engineering", "hard"),
        ]
    ]
    dataset = DatasetDict({"train": Dataset.from_list(rows)})
    monkeypatch.setattr(lighteval_task, "load_dataset", lambda *args, **kwargs: dataset)
    task = LightevalTask(CONFIGS["supergpqa:engineering:hard:cf"])
    for docs in (task.eval_docs(), task.fewshot_docs()):
        assert len(docs) == 1
        assert docs[0].query == "Question: Selected question\nAnswer:"


def test_inspect_filters_fewshots_before_limiting(monkeypatch):
    from lighteval import main_inspect

    rows = [
        {
            "question": question,
            "options": ["Yes", "No"],
            "answer_letter": "A",
            "discipline": discipline,
            "difficulty": difficulty,
        }
        for question, discipline, difficulty in [
            ("Wrong discipline", "Science", "hard"),
            ("Wrong difficulty", "Engineering", "easy"),
            ("Selected question", "Engineering", "hard"),
            ("Second selected question", "Engineering", "hard"),
        ]
    ]
    limits = []
    prompts = []
    original_system_message = main_inspect.system_message

    def fake_hf_dataset(*args, sample_fields, limit=None, **kwargs):
        limits.append(limit)
        samples = MemoryDataset([sample_fields(row) for row in rows])
        return samples[:limit] if limit is not None else samples

    def capture_system_message(prompt):
        prompts.append(prompt)
        return original_system_message(prompt)

    monkeypatch.setattr(main_inspect, "hf_dataset", fake_hf_dataset)
    monkeypatch.setattr(main_inspect, "system_message", capture_system_message)
    config = replace(deepcopy(CONFIGS["supergpqa:engineering:hard:cot"]), num_fewshots=1)
    task = main_inspect.get_inspect_ai_task(config)
    assert [sample.input for sample in task.dataset] == ["Second selected question"]
    assert limits == [None, None]
    assert prompts == ["Selected question\n\nA) Yes\nB) No\n\nANSWER: A"]


@pytest.mark.parametrize("selection", ["mmlu:cf", "mmlu:abstract_algebra"])
def test_nested_groups_preserve_mmlu_selection(monkeypatch, selection):
    configs = {config.name: config for config in mmlu.TASKS_TABLE}
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: configs))
    registry = Registry(f"{selection}|0")
    if selection == "mmlu:cf":
        expected = {name for name in configs if name.startswith("mmlu:") and name.endswith(":cf")}
        assert len(expected) == 57
    else:
        expected = {f"mmlu:abstract_algebra:{variant}" for variant in ("cf", "mcf", "mcf_em", "chat")}
    assert set(registry.task_to_configs) == expected
