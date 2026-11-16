from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai.dataset import MemoryDataset

from lighteval.models.model_output import ModelResponse
from lighteval.tasks import lighteval_task
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.prompt_manager import PromptManager
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
# 13 disciplines x 4 variants (all difficulties each) + the 13 x 3 `mcf_em` difficulty cells (owner 11-16: SuperGPQA easy / middle / hard)
DISCIPLINE_NAMES = {f"supergpqa:{discipline}:{variant}" for discipline in DISCIPLINES for variant in VARIANTS}
CELL_NAMES = {f"supergpqa:{discipline}:{difficulty}:mcf_em" for discipline in DISCIPLINES for difficulty in DIFFICULTIES}
EXPECTED_NAMES = DISCIPLINE_NAMES | CELL_NAMES


def make_record(n_options: int) -> dict:
    return {
        "uuid": f"uuid-{n_options}",
        "question": " Which option is correct? ",
        "discipline": "Engineering",
        "difficulty": "hard",
        "options": [f"Option {i}" for i in range(n_options)],
        # The dataset's answer text can differ in capitalization from its option.
        "answer": f"option {n_options - 1}",
        "answer_letter": {4: "D", 6: "F", 10: "J"}[n_options],
    }


N_OPTIONS = [4, 6, 10]


@pytest.mark.parametrize("n_options", N_OPTIONS)
def test_prompts_preserve_options_and_use_answer_letter(n_options):
    record = make_record(n_options)
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


@pytest.mark.parametrize("n_options", N_OPTIONS)
@pytest.mark.parametrize("prompt", [supergpqa_cf_prompt, supergpqa_mcf_prompt, supergpqa_cot_prompt])
def test_first_option_can_be_correct(n_options, prompt):
    record = make_record(n_options)
    record["answer_letter"] = "A"
    assert prompt(record).gold_index == 0


@pytest.mark.parametrize("n_options", N_OPTIONS)
def test_inspect_sample_and_fewshot_use_letter_target(n_options):
    record = make_record(n_options)
    sample = record_to_sample(record)
    assert sample.input == record["question"].strip()
    assert sample.choices == record["options"]
    assert sample.target == record["answer_letter"]
    assert sample.metadata == {"discipline": "Engineering", "difficulty": "hard"}
    fewshot = sample_to_fewshot(sample)
    assert f"{sample.target}) {record['options'][-1]}" in fewshot
    assert fewshot.endswith(f"ANSWER: {sample.target}")


@pytest.mark.parametrize("n_options", N_OPTIONS)
@pytest.mark.parametrize("correct", [True, False])
def test_reasoning_metric_scores_all_supported_answer_letters(n_options, correct):
    record = make_record(n_options)
    doc = supergpqa_cot_prompt(record)
    answer = record["answer_letter"] if correct else "A"
    response = ModelResponse(text=[f"After considering the options, I can conclude.\nAnswer: {answer}"])
    score = supergpqa_cot.metrics[0].sample_level_fn.compute(doc, response)
    assert score == float(correct)


@pytest.mark.parametrize("n_options", N_OPTIONS)
@pytest.mark.parametrize("correct", [True, False])
def test_exact_match_metric_scores_answer_letters(n_options, correct):
    record = make_record(n_options)
    doc = supergpqa_mcf_prompt(record)
    answer = record["answer_letter"] if correct else "A"
    response = ModelResponse(text=[f" {answer}"])
    score = supergpqa_mcf_em.metrics[0].sample_level_fn.compute(doc, response)
    assert score == float(correct)


def test_supergpqa_module_discovery_and_selection(monkeypatch):
    expected_names = EXPECTED_NAMES
    assert len(TASKS_TABLE) == len(expected_names) == 52 + 39
    configs = Registry._load_from_files([Path(supergpqa.__file__)], "lighteval.tasks.tasks")
    assert set(configs) == expected_names
    # Keep this unit test independent of unrelated tasks and optional judge backends.
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: configs))
    # The bare family `supergpqa` is the 52 discipline tasks: the difficulty cells belong to their own supersets only (`superset_keys`).
    registry = Registry("supergpqa|0")
    assert set(registry.task_to_configs) == DISCIPLINE_NAMES
    assert set(registry.load_tasks()) == {f"{name}|0" for name in DISCIPLINE_NAMES}
    for config in TASKS_TABLE:
        assert config.hf_repo == "m-a-p/SuperGPQA"
        assert config.hf_subset == "default"
        assert config.hf_avail_splits == ("train",)
        assert config.evaluation_splits == ("train",)


@pytest.mark.parametrize(
    ("selection", "expected_names"),
    [
        ("supergpqa:cf", {name for name in DISCIPLINE_NAMES if name.endswith(":cf")}),
        ("supergpqa:engineering", {name for name in DISCIPLINE_NAMES if name.startswith("supergpqa:engineering:")}),
        ("supergpqa:engineering:cf", {"supergpqa:engineering:cf"}),
        # `supergpqa:mcf_em` (the old task) stays the 13 discipline tasks: the 39 difficulty cells are not in it (owner 11-16)
        ("supergpqa:mcf_em", {f"supergpqa:{d}:mcf_em" for d in DISCIPLINES}),
        ("supergpqa:easy:mcf_em", {f"supergpqa:{d}:easy:mcf_em" for d in DISCIPLINES}),
        ("supergpqa:middle:mcf_em", {f"supergpqa:{d}:middle:mcf_em" for d in DISCIPLINES}),
        ("supergpqa:hard:mcf_em", {f"supergpqa:{d}:hard:mcf_em" for d in DISCIPLINES}),
        ("supergpqa:engineering:hard:mcf_em", {"supergpqa:engineering:hard:mcf_em"}),
    ],
)
def test_select_subtask_groups(monkeypatch, selection, expected_names):
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: CONFIGS))
    registry = Registry(f"{selection}|2")
    assert set(registry.task_to_configs) == expected_names
    assert set(registry.load_tasks()) == {f"{name}|2" for name in expected_names}


def test_difficulty_cells_have_the_settings_of_the_discipline_task():
    for discipline in DISCIPLINES:
        base = CONFIGS[f"supergpqa:{discipline}:mcf_em"]
        assert base.eval_doc_filter is None and base.superset_keys is None
        for difficulty in DIFFICULTIES:
            cell = CONFIGS[f"supergpqa:{discipline}:{difficulty}:mcf_em"]
            assert cell.superset_keys == (f"supergpqa:{difficulty}:mcf_em",) and cell.eval_doc_filter is not None
            # the same prompt function, few-shot rule, generation and scoring; the same few-shot pool (hf_filter = the discipline's rows of every difficulty)
            for field in ("prompt_function", "few_shots_split", "few_shots_select", "generation_size", "stop_sequence", "metrics", "hf_repo", "hf_revision", "hf_subset", "evaluation_splits"):
                assert getattr(cell, field) == getattr(base, field), (cell.name, field)
            row = {"discipline": DISCIPLINES[discipline], "difficulty": difficulty}
            assert cell.hf_filter(row) and cell.hf_filter({**row, "difficulty": "other"}) and not cell.hf_filter({**row, "discipline": "Other"})
            assert cell.hf_filter.func is base.hf_filter.func and cell.hf_filter.keywords == base.hf_filter.keywords


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
    # the questions each config's `hf_filter` (the few-shot pool, with the evaluation docs) and `filter` (the inspect path's samples) select
    all_levels = [f"{DISCIPLINES[discipline]} {level}" for level in DIFFICULTIES]
    expected = all_levels if difficulty is None else [f"{DISCIPLINES[discipline]} {difficulty}"]
    for variant in VARIANTS if difficulty is None else ("mcf_em",):
        suffix = f":{difficulty}" if difficulty else ""
        config = CONFIGS[f"supergpqa:{discipline}{suffix}:{variant}"]
        # a difficulty cell draws its shots from the whole discipline and keeps its difficulty afterwards (eval_doc_filter, see the next tests)
        assert [row["question"] for row in rows if config.hf_filter(row)] == all_levels
        samples = MemoryDataset([config.sample_fields(row) for row in rows]).filter(config.filter)
        assert [sample.input for sample in samples] == expected


def synthetic_rows(per_cell: int = 8) -> list[dict]:
    """per_cell questions for every (discipline, difficulty) of two disciplines; 4 options, unique uuid / question."""
    return [
        {
            "uuid": f"u-{discipline}-{level}-{k}",
            "question": f"Question {discipline} {level} {k}",
            "options": ["Yes", "No", "Maybe", "Never"],
            "answer_letter": "ABCD"[k % 4],
            "discipline": discipline,
            "difficulty": level,
        }
        for discipline in ("Engineering", "Science")
        for level in DIFFICULTIES
        for k in range(per_cell)
    ]


def docs_of(monkeypatch, name: str, rows: list[dict], shots: int = 5) -> list:
    dataset = DatasetDict({"train": Dataset.from_list(rows)})
    monkeypatch.setattr(lighteval_task, "load_dataset", lambda *args, **kwargs: dataset)
    task = LightevalTask(replace(CONFIGS[name], num_fewshots=shots))
    return task.get_docs()


def test_lighteval_filters_evaluation_and_fewshot_rows(monkeypatch):
    rows = [
        {
            "uuid": question,
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
    # the discipline task: both Engineering questions are evaluated and are each other's shots
    task = LightevalTask(replace(CONFIGS["supergpqa:engineering:mcf_em"], num_fewshots=1))
    assert [doc.specific["uuid"] for doc in task.fewshot_docs()] == ["Wrong difficulty", "Selected question"]
    assert sorted(doc.specific["uuid"] for doc in task.get_docs()) == ["Selected question", "Wrong difficulty"]
    # the cell: the shots still come from the whole discipline, only the hard question is evaluated; it has the other question as its shot
    task = LightevalTask(replace(CONFIGS["supergpqa:engineering:hard:mcf_em"], num_fewshots=1))
    assert [doc.specific["uuid"] for doc in task.fewshot_docs()] == ["Wrong difficulty", "Selected question"]
    docs = task.get_docs()
    assert [doc.specific["uuid"] for doc in docs] == ["Selected question"]
    assert [shot.specific["uuid"] for shot in docs[0].fewshot_samples] == ["Wrong difficulty"]
    assert docs[0].query == "Answer the following multiple choice question.\n\nQuestion: Selected question\nA: Yes\nB: No\nAnswer:"
    # `max_samples` counts the cell's own questions
    assert [doc.specific["uuid"] for doc in LightevalTask(replace(CONFIGS["supergpqa:engineering:hard:mcf_em"], num_fewshots=1)).get_docs(1)] == ["Selected question"]
    # a cell without any question is an error, not an empty task
    monkeypatch.setattr(lighteval_task, "load_dataset", lambda *args, **kwargs: DatasetDict({"train": Dataset.from_list(rows[:2])}))
    try:
        LightevalTask(replace(CONFIGS["supergpqa:engineering:hard:mcf_em"], num_fewshots=1)).get_docs()
    except ValueError as error:
        assert "no documents" in str(error)
    else:
        raise AssertionError("a cell without questions must raise")


@pytest.mark.parametrize("discipline", ["engineering", "science"])
def test_difficulty_cells_build_the_prompts_of_the_discipline_task(monkeypatch, discipline):
    """Owner 11-16 (three tasks easy / middle / hard): every question gets the byte-identical prompt (the same shots in the same order) in its cell as in `supergpqa:<discipline>:mcf_em`."""
    rows = synthetic_rows()
    manager = PromptManager(False)
    base = {doc.specific["uuid"]: doc for doc in docs_of(monkeypatch, f"supergpqa:{discipline}:mcf_em", rows)}
    assert len(base) == 3 * 8 and all(len(doc.fewshot_samples) == 5 for doc in base.values())
    assert len({manager.prepare_prompt(doc) for doc in base.values()}) == len(base)
    seen = {}
    for difficulty in DIFFICULTIES:
        docs = docs_of(monkeypatch, f"supergpqa:{discipline}:{difficulty}:mcf_em", rows)
        assert len(docs) == 8 and {doc.specific["difficulty"] for doc in docs} == {difficulty}
        for doc in docs:
            uuid = doc.specific["uuid"]
            assert manager.prepare_prompt(doc) == manager.prepare_prompt(base[uuid]), uuid
            assert [s.specific["uuid"] for s in doc.fewshot_samples] == [s.specific["uuid"] for s in base[uuid].fewshot_samples]
            assert doc.gold_index == base[uuid].gold_index and doc.choices == base[uuid].choices
            assert doc.generation_size == 5 and list(doc.stop_sequences) == ["\n"]
            seen[uuid] = difficulty
    assert set(seen) == set(base)                                    # the three cells together are the discipline task: every question once


def test_eval_doc_filter_keeps_the_random_stream_of_the_unfiltered_task(monkeypatch):
    rows = synthetic_rows()
    plain = {doc.specific["uuid"]: doc for doc in docs_of(monkeypatch, "supergpqa:science:mcf_em", rows)}
    docs = docs_of(monkeypatch, "supergpqa:science:middle:mcf_em", rows)
    # the shots of the cell's first question are those of the discipline task, drawn after the questions before it in the shuffled order (a cell-only stream would differ)
    assert [d.specific["uuid"] for d in docs] == [u for u in plain if plain[u].specific["difficulty"] == "middle"]
    cell_only = replace(CONFIGS["supergpqa:science:middle:mcf_em"], eval_doc_filter=None, hf_filter=lambda row: row["discipline"] == "Science" and row["difficulty"] == "middle", num_fewshots=5)
    dataset = DatasetDict({"train": Dataset.from_list(rows)})
    monkeypatch.setattr(lighteval_task, "load_dataset", lambda *args, **kwargs: dataset)
    other = LightevalTask(cell_only).get_docs()
    manager = PromptManager(False)
    assert {manager.prepare_prompt(d) for d in other} != {manager.prepare_prompt(d) for d in docs}


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
    # the inspect path has no `eval_doc_filter`: the `filter` of a cell selects its discipline and difficulty, for the questions and for the shots
    config = replace(deepcopy(CONFIGS["supergpqa:engineering:hard:mcf_em"]), num_fewshots=1)
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
