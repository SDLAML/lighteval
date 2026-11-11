import asyncio
from copy import deepcopy
from dataclasses import replace
from string import ascii_uppercase

import pytest
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks import agieval


CONFIGS = {config.hf_subset: config for config in agieval.TASKS_TABLE if config.name.endswith(":cot")}
MCF_CONFIGS = {config.hf_subset: config for config in agieval.TASKS_TABLE if config.name.endswith(":mcf")}


@pytest.fixture(params=[4, 5])
def record(request):
    return {
        "passage": " A tower is 100 metres tall. ",
        "question": " Which statement follows from the passage? ",
        "options": [f"({label}) Option {label} refers to (A)." for label in ascii_uppercase[: request.param]],
        "label": ascii_uppercase[request.param - 1],
        "answer": "",
    }


def test_both_backends_preserve_context_options_and_labels(record):
    original = deepcopy(record)
    sample = agieval.record_to_sample(record)
    doc = agieval.agieval_prompt_cot_eng(record, "agieval_eng:test:cot")
    assert (
        sample.input == "Passage:\nA tower is 100 metres tall.\n\nQuestion: Which statement follows from the passage?"
    )
    assert sample.input in doc.query
    assert sample.target == record["label"]
    assert doc.get_golds() == [f" {record['label']}"]
    assert doc.task_name == "agieval_eng:test:cot"
    assert f"LETTER is one of {ascii_uppercase[: len(record['options'])]}." in doc.instruction
    for label, option in zip(ascii_uppercase, sample.choices):
        assert option == f"Option {label} refers to (A)."
        assert f"\n{label}. {option}" in doc.query
    assert record == original


@pytest.mark.parametrize("passage", [None, "", "   "])
def test_optional_passage_does_not_add_empty_context(record, passage):
    record["passage"] = passage
    sample = agieval.record_to_sample(record)
    assert sample.input == "Question: Which statement follows from the passage?"
    assert "Passage:" not in agieval.agieval_prompt_cot_eng(record).query


@pytest.mark.parametrize("prefix", ["A ", "A) ", "A. ", "(A)", " (A) "])
def test_only_the_leading_option_label_is_removed(record, prefix):
    record["options"][0] = prefix + "Keep (B) inside the answer."
    assert agieval.record_to_sample(record).choices[0] == "Keep (B) inside the answer."


@pytest.mark.parametrize("label", ["", "Z", "AB"])
def test_invalid_answer_letters_are_rejected(record, label):
    record["label"] = label
    with pytest.raises(ValueError, match="valid answer letter"):
        agieval.record_to_sample(record)


def test_metric_scores_all_explicit_answer_letters(record):
    metric = CONFIGS["aqua_rat"].metrics[0].sample_level_fn
    for label in ascii_uppercase[: len(record["options"])]:
        record["label"] = label
        doc = agieval.agieval_prompt_cot_eng(record)
        assert metric.compute(doc, ModelResponse(text=[f"A. Consider the first option.\nAnswer: {label}"])) == 1
        wrong = "B" if label == "A" else "A"
        assert metric.compute(doc, ModelResponse(text=[f"Answer: {wrong}"])) == 0


def test_metric_uses_postprocessed_final_text(record):
    doc = agieval.agieval_prompt_cot_eng(record)
    response = ModelResponse(text=["Answer: A"], text_post_processed=[f"Answer: {record['label']}"])
    assert CONFIGS["aqua_rat"].metrics[0].sample_level_fn.compute(doc, response) == 1


def test_fewshot_examples_demonstrate_the_requested_answer_format(record):
    example = agieval.agieval_prompt_cot_eng({**record, "__few_shots": True})
    doc = agieval.agieval_prompt_cot_eng(record)
    doc.fewshot_samples = [example]
    prompt = PromptManager().prepare_prompt(doc)
    assert f"\n\nAnswer: {record['label']}" in prompt
    assert "Solution:" not in example.query
    sample = agieval.record_to_sample(record)
    inspect_example = agieval.sample_to_fewshot(sample)
    assert sample.input in inspect_example
    assert inspect_example.endswith(f"Answer: {record['label']}")


def test_mcf_scores_answer_letters_without_generation(record):
    config = replace(MCF_CONFIGS["aqua_rat"], num_fewshots=1)
    task = LightevalTask(config)
    task.dataset = {
        "train": [record],
        "validation": [{**record, "question": "An example question."}],
    }
    doc = task.get_docs()[0]
    assert doc.sampling_methods == [SamplingMethod.LOGPROBS]
    assert doc.generation_size == -1
    assert doc.choices == [f" {label}" for label in ascii_uppercase[: len(record["options"])]]
    assert "Passage:\nA tower is 100 metres tall." in doc.query
    for label in ascii_uppercase[: len(record["options"])]:
        assert f"\n{label}. Option {label} refers to (A)." in doc.query
    prompt = PromptManager().prepare_prompt(doc)
    assert prompt.count(doc.instruction.strip()) == 1
    assert f"Answer: {record['label']}" in prompt
    assert prompt.endswith("Answer:")
    for predicted_index in (0, doc.gold_index):
        logprobs = [-10.0] * len(doc.choices)
        logprobs[predicted_index] = -0.1
        for metric in config.metrics:
            assert metric.sample_level_fn.compute(doc, ModelResponse(logprobs=logprobs)) == int(
                predicted_index == doc.gold_index
            )


@pytest.mark.parametrize("configs", [CONFIGS, MCF_CONFIGS], ids=["cot", "mcf"])
def test_logiqa_fewshots_use_real_logiqa_and_exclude_current_example(record, configs):
    config = replace(configs["logiqa-en"], num_fewshots=1)
    task = LightevalTask(config)
    task.dataset = {
        "train": [{**record, "passage": f"LogiQA context {i}"} for i in range(3)],
        "validation": [{**record, "passage": "", "question": "An unrelated AQUA math question"}],
    }
    docs = task.get_docs()
    assert len(docs) == 3
    for doc in docs:
        assert len(doc.fewshot_samples) == 1
        example = doc.fewshot_samples[0]
        assert "LogiQA context" in example.query
        assert "AQUA" not in example.query
        assert doc.source_id != example.source_id
        assert doc.query != example.query


@pytest.mark.parametrize("subset", ["logiqa-en", "sat-en"])
def test_inspect_fewshot_split_and_demonstration_exclusion(monkeypatch, record, subset):
    from lighteval import main_inspect

    rows = {
        "train": [{**record, "passage": f"Context {i}"} for i in range(3)],
        "validation": [{**record, "passage": "Validation context"}],
    }
    splits = []
    prompts = []
    original_system_message = main_inspect.system_message

    def fake_hf_dataset(*args, split, sample_fields, limit=None, **kwargs):
        splits.append(split)
        samples = MemoryDataset([sample_fields(row) for row in rows[split]])
        return samples[:limit] if limit is not None else samples

    def capture_system_message(prompt):
        prompts.append(prompt)
        return original_system_message(prompt)

    monkeypatch.setattr(main_inspect, "hf_dataset", fake_hf_dataset)
    monkeypatch.setattr(main_inspect, "system_message", capture_system_message)
    config = replace(CONFIGS[subset], num_fewshots=1)
    original_solvers = list(config.solver)
    for _ in range(2):
        task = main_inspect.get_inspect_ai_task(config)
        if subset == "logiqa-en":
            assert len(task.dataset) == 2
            assert all("Context 0" not in sample.input for sample in task.dataset)
        else:
            assert len(task.dataset) == 3
            assert "Validation context" in prompts[-1]
        assert prompts[-1].endswith(f"Answer: {record['label']}")
        assert config.solver == original_solvers
    expected_split = "train" if subset == "logiqa-en" else "validation"
    assert splits == ["train", expected_split, "train", expected_split]


def test_inspect_solver_requests_reasoning_and_scores_letter_targets(record):
    config = CONFIGS["aqua_rat"]
    sample = config.sample_fields(record)
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="example",
        epoch=1,
        input=sample.input,
        messages=[ChatMessageUser(content=sample.input)],
        choices=sample.choices,
    )

    async def fake_generate(state, **kwargs):
        assert "Think step by step" in state.user_prompt.text
        assert "A tower is 100 metres tall." in state.user_prompt.text
        assert kwargs["max_tokens"] == config.generation_size
        state.output = ModelOutput.from_content("mockllm/model", f"A. Consider this option.\nANSWER: {sample.target}")
        return state

    state = asyncio.run(config.solver[0](state, fake_generate))
    score = asyncio.run(config.scorer(state, Target(sample.target)))
    assert score.value == "C"


@pytest.mark.parametrize("target", list("ABCD"))
@pytest.mark.parametrize("prefix", ["({label})", "{label} ", "{label}) ", "{label}. "])
def test_out_of_order_options_keep_original_label_meanings(record, target, prefix):
    record["options"] = [
        prefix.format(label=label) + f"Option {label} refers to (A)."
        for label in ascii_uppercase[: len(record["options"])]
    ]
    record["options"][1], record["options"][2] = record["options"][2], record["options"][1]
    record["label"] = target
    original = deepcopy(record)
    sample = agieval.record_to_sample(record)
    assert sample.target == target
    assert sample.choices[ascii_uppercase.index(target)] == f"Option {target} refers to (A)."
    for prompt_function in (agieval.agieval_prompt_cot_eng, agieval.agieval_mcf_prompt):
        doc = prompt_function(record)
        assert doc.get_golds() == [f" {target}"]
        assert "B. Option B refers to (A).\nC. Option C refers to (A)." in doc.query
        assert "B. (C)" not in doc.query
        assert "C. (B)" not in doc.query
    assert agieval.sample_to_fewshot(sample).endswith(f"Answer: {target}")
    assert record == original


def test_duplicate_option_labels_are_rejected(record):
    record["options"][2] = "(B)A second option with the same label."
    with pytest.raises(ValueError, match="option labels must be unique"):
        agieval.record_to_sample(record)


@pytest.fixture
def conflicting_sat_record():
    return {
        "passage": (
            "At last, Old Widow Lau was done haggling with the driver "
            "and we stepped inside Father's shop."
        ),
        "question": "Throughout the passage, the narrator is portrayed as someone who is",
        "options": [
            "(A)reserved around unfamiliar people.",
            "(B)attuned to her immediate surroundings.",
            "(C)sympathetic to the needs of others.",
            "(D)anxious about her responsibilities.",
        ],
        "label": "A",
        "answer": "",
    }


@pytest.mark.parametrize("variant", ["cot", "mcf", "mcf_em"])
def test_sat_conflicts_are_excluded_from_native_evaluation_and_fewshots(record, conflicting_sat_record, variant):
    config = next(config for config in agieval.TASKS_TABLE if config.name == f"agieval_eng:sat-en:{variant}")
    task = LightevalTask(replace(config, num_fewshots=1))
    conflicts = [{**conflicting_sat_record, "label": label} for label in "AB"]
    task.dataset = {
        "train": [conflicts[0], record, conflicts[1]],
        "validation": [conflicts[1], {**record, "question": "An example question."}, conflicts[0]],
    }
    docs = task.get_docs()
    assert len(docs) == 1
    assert "Which statement follows from the passage?" in docs[0].query
    assert len(docs[0].fewshot_samples) == 1
    assert "An example question." in docs[0].fewshot_samples[0].query


def test_sat_conflicts_are_excluded_from_inspect_evaluation_and_fewshots(monkeypatch, record, conflicting_sat_record):
    from lighteval import main_inspect

    conflicts = [{**conflicting_sat_record, "label": label} for label in "AB"]
    rows = {
        "train": [conflicts[0], record, conflicts[1]],
        "validation": [conflicts[1], {**record, "question": "An example question."}, conflicts[0]],
    }
    limits = []
    prompts = []
    original_system_message = main_inspect.system_message

    def fake_hf_dataset(*args, split, sample_fields, limit=None, **kwargs):
        limits.append(limit)
        samples = MemoryDataset([sample_fields(row) for row in rows[split]])
        return samples[:limit] if limit is not None else samples

    def capture_system_message(prompt):
        prompts.append(prompt)
        return original_system_message(prompt)

    monkeypatch.setattr(main_inspect, "hf_dataset", fake_hf_dataset)
    monkeypatch.setattr(main_inspect, "system_message", capture_system_message)
    task = main_inspect.get_inspect_ai_task(replace(CONFIGS["sat-en"], num_fewshots=1))
    assert len(task.dataset) == 1
    assert "Which statement follows from the passage?" in task.dataset[0].input
    assert "An example question." in prompts[0]
    assert "Throughout the passage" not in prompts[0]
    assert limits == [None, None]


@pytest.mark.parametrize("field", ["passage", "question", "options"])
def test_sat_conflict_filter_preserves_different_content(conflicting_sat_record, field):
    if field == "options":
        conflicting_sat_record[field][0] = "(A)A different option."
    else:
        conflicting_sat_record[field] = "Different content."
    sample = agieval.record_to_sample(conflicting_sat_record)
    assert CONFIGS["sat-en"].filter(sample)
    assert agieval.agieval_prompt_cot_eng(conflicting_sat_record) is not None
    assert agieval.agieval_mcf_prompt(conflicting_sat_record) is not None
