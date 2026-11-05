from copy import deepcopy

import pytest
from datasets import Dataset, DatasetDict

from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.tasks.mmlu_cf import TASKS_TABLE


CATEGORIES = (
    "Biology",
    "Business",
    "Chemistry",
    "Computer_Science",
    "Economics",
    "Engineering",
    "Health",
    "History",
    "Law",
    "Math",
    "Other",
    "Philosophy",
    "Physics",
    "Psychology",
)
VARIANTS = ("cf", "mcf", "mcf_em", "chat")
CONFIGS = {config.name: config for config in TASKS_TABLE}


@pytest.fixture
def record():
    return {
        "Question": " Which option is correct? ",
        "A": " First option ",
        "B": "Second option",
        "C": " Third option",
        "D": "Fourth option ",
        "Answer": "C",
    }


def test_registers_all_categories_and_variants():
    expected = {f"mmlu_cf:{category.lower()}:{variant}" for category in CATEGORIES for variant in VARIANTS}
    assert set(CONFIGS) == expected
    assert len(TASKS_TABLE) == len(expected)
    for category in CATEGORIES:
        for variant in VARIANTS:
            config = CONFIGS[f"mmlu_cf:{category.lower()}:{variant}"]
            assert config.hf_repo == "microsoft/MMLU-CF"
            assert config.hf_subset == "default"
            assert config.evaluation_splits == (f"{category}_val",)
            assert config.few_shots_split == f"{category}_dev"


@pytest.mark.parametrize("answer", list("ABCD"))
@pytest.mark.parametrize("variant", VARIANTS)
def test_prompts_preserve_choice_order_and_gold(record, answer, variant):
    record["Answer"] = answer
    original = deepcopy(record)
    config = CONFIGS[f"mmlu_cf:computer_science:{variant}"]
    doc = config.prompt_function(record, config.name)

    assert record == original
    assert doc.task_name == config.name
    assert "about computer science." in doc.instruction
    assert "Which option is correct?" in doc.query
    assert doc.gold_index == "ABCD".index(answer)
    assert doc.fewshot_sorting_class == record[answer].strip()
    if variant == "cf":
        assert doc.choices == [f" {record[label].strip()}" for label in "ABCD"]
        assert doc.get_golds() == [f" {record[answer].strip()}"]
        assert "First option" not in doc.query
    else:
        assert doc.choices == [" A", " B", " C", " D"]
        assert doc.get_golds() == [f" {answer}"]
        for label in "ABCD":
            assert f"{label}. {record[label].strip()}" in doc.query
    if variant == "chat":
        assert doc.query.endswith("Think step by step before answering.")
    else:
        assert doc.query.endswith("\nAnswer:")


@pytest.mark.parametrize("category", CATEGORIES)
def test_evaluation_and_fewshots_use_matching_category_splits(record, category):
    config = CONFIGS[f"mmlu_cf:{category.lower()}:mcf"]
    task = LightevalTask(config)
    task.dataset = DatasetDict(
        {
            f"{subject}_{split}": Dataset.from_list([{**record, "Question": f"{subject} {split}"}])
            for subject in CATEGORIES
            for split in ("val", "dev")
        }
    )

    eval_docs = task.eval_docs()
    fewshot_docs = task.fewshot_docs()
    assert len(eval_docs) == len(fewshot_docs) == 1
    assert f"Question: {category} val\n" in eval_docs[0].query
    assert f"Question: {category} dev\n" in fewshot_docs[0].query


@pytest.mark.parametrize("selection", ["mmlu_cf", "mmlu_cf:cf", "mmlu_cf:computer_science"])
def test_group_selectors(monkeypatch, selection):
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: CONFIGS))
    registry = Registry(f"{selection}|5")
    if selection == "mmlu_cf:cf":
        expected = {f"mmlu_cf:{category.lower()}:cf" for category in CATEGORIES}
    elif selection == "mmlu_cf:computer_science":
        expected = {f"mmlu_cf:computer_science:{variant}" for variant in VARIANTS}
    else:
        expected = set(CONFIGS)
    assert set(registry.task_to_configs) == expected
    assert all(config.num_fewshots == 5 for configs in registry.task_to_configs.values() for config in configs)
