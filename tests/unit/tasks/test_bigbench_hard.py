import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from string import ascii_uppercase

import pytest

from lighteval.metrics import apply_metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks import bigbench_hard
from lighteval.tasks.tasks.bigbench_hard import (
    TASKS_TABLE,
    bbh_chat_prompt,
    bbh_mcf_prompt,
    record_to_sample,
    sample_to_fewshot,
)


MCF_SUBSETS = {
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
}
YES_NO_SUBSETS = {"causal_judgement", "navigate", "sports_understanding", "web_of_lies"}
BINARY_CHOICES = {
    **dict.fromkeys(YES_NO_SUBSETS, ("Yes", "No")),
    "boolean_expressions": ("True", "False"),
    "formal_fallacies": ("valid", "invalid"),
}
BINARY_SUBSETS = set(BINARY_CHOICES)
CONFIGS = {config.name: config for config in TASKS_TABLE}


@pytest.fixture(params=[2, 3, 4, 6, 11, 18])
def record(request):
    labels = ascii_uppercase[: request.param]
    options = "\n".join(f"({label}) Option {label}" for label in labels)
    return {"input": f"Which option follows (A) in the question?\nOptions:\n{options}", "target": f"({labels[-1]})"}


def test_prompts_and_inspect_preserve_options_and_gold(record):
    original = deepcopy(record)
    sample = record_to_sample(record)
    labels = ascii_uppercase[: len(sample.choices)]
    assert sample.input == "Which option follows (A) in the question?"
    assert sample.choices == [f"Option {label}" for label in labels]
    assert sample.target == labels[-1]
    assert sample_to_fewshot(sample).endswith(f"{labels[-1]}) Option {labels[-1]}\n\nANSWER: {labels[-1]}")
    for prompt in (bbh_mcf_prompt, bbh_chat_prompt):
        doc = prompt(record, task_name="bbh_test")
        assert doc.task_name == "bbh_test"
        assert doc.choices == [f" {label}" for label in labels]
        assert doc.get_golds() == [f" {labels[-1]}"]
        assert doc.gold_index == len(labels) - 1
        for label in labels:
            assert f"\n{label}: Option {label}" in doc.query
        assert "Options:" not in doc.query
    assert f"LETTER is one of {labels}." in bbh_chat_prompt(record).instruction
    assert record == original


def test_multiline_options_and_first_answer_are_preserved():
    record = {"input": "Question?\nOptions:\n(A) First line\nsecond line\n(B) Other", "target": "(A)"}
    assert record_to_sample(record).choices == ["First line\nsecond line", "Other"]
    assert bbh_mcf_prompt(record).gold_index == 0


@pytest.mark.parametrize("target", ["Yes", "False", "4", "Monsters, Inc", "dearth, wind, & fire"])
def test_nonletter_answers_are_excluded_from_both_backends(record, target):
    record["target"] = target
    assert bbh_mcf_prompt(record) is None
    assert bbh_chat_prompt(record) is None
    assert record_to_sample(record) == []


@pytest.mark.parametrize(
    "record",
    [
        {"input": "Question without options", "target": "(A)"},
        {"input": "Question?\nOptions:\n(A) One\n(C) Three", "target": "(C)"},
        {"input": "Question?\nOptions:\n(A) One\n(B) Two", "target": "(C)"},
    ],
)
def test_invalid_letter_records_fail_explicitly(record):
    with pytest.raises(ValueError, match="consecutive labels containing the gold"):
        bbh_mcf_prompt(record)


@pytest.mark.parametrize("variant", ["mcf", "mcf_em"])
@pytest.mark.parametrize("correct", [True, False])
def test_mcf_metric_dispatch_scores_labels(record, correct, variant):
    config = CONFIGS[f"bigbench_hard:date_understanding:{variant}"]
    doc = config.prompt_function(record)
    predicted_index = doc.gold_index if correct else 0
    if variant == "mcf_em":
        response = ModelResponse(text=[doc.choices[predicted_index]])
        expected = {"em": float(correct)}
    else:
        logprobs = [-10.0] * len(doc.choices)
        logprobs[predicted_index] = -0.1
        response = ModelResponse(logprobs=logprobs)
        expected = {"acc": int(correct), "acc_norm": int(correct)}
    assert apply_metric([response], [doc], config.metrics) == [expected]


@pytest.mark.parametrize("correct", [True, False])
def test_chat_metric_dispatch_scores_letters_beyond_d(record, correct):
    doc = bbh_chat_prompt(record)
    answer = record["target"][1] if correct else "A"
    response = ModelResponse(
        text=["Unprocessed reasoning"],
        text_post_processed=[f"After considering the options, I can conclude.\nAnswer: {answer}"],
    )
    config = CONFIGS["bigbench_hard:date_understanding:chat"]
    assert apply_metric([response], [doc], config.metrics) == [{"extractive_match": float(correct)}]


def test_fewshots_exclude_the_evaluated_row_and_skip_nonletter_records():
    config = replace(CONFIGS["bigbench_hard:hyperbaton:chat"], num_fewshots=1)
    task = LightevalTask(config)
    task.dataset = {
        "test": [{"input": f"Question {i}?\nOptions:\n(A) First\n(B) Second", "target": "(B)"} for i in range(2)]
        + [{"input": "Malformed options", "target": "Text answer"}]
    }
    docs = task.get_docs()
    assert len(docs) == len(task.fewshot_docs()) == 2
    for doc in docs:
        assert len(doc.fewshot_samples) == 1
        assert doc.fewshot_samples[0].source_id != doc.source_id
        rendered = PromptManager().prepare_prompt(doc)
        assert "Think step by step before answering." in rendered
        assert rendered.count("A: First\nB: Second") == 2
        assert "Malformed options" not in rendered


def test_module_discovery_registers_only_multiple_choice_subtasks():
    configs = Registry._load_from_files([Path(bigbench_hard.__file__)], "lighteval.tasks.tasks")
    expected = {
        f"bigbench_hard:{subset}:{variant}"
        for subset in MCF_SUBSETS | BINARY_SUBSETS
        for variant in ("mcf", "mcf_em", "chat")
    }
    assert len(TASKS_TABLE) == len(configs) == len(expected) == 69
    assert set(configs) == expected
    for subset in MCF_SUBSETS | BINARY_SUBSETS:
        mcf = configs[f"bigbench_hard:{subset}:mcf"]
        mcf_em = configs[f"bigbench_hard:{subset}:mcf_em"]
        chat = configs[f"bigbench_hard:{subset}:chat"]
        assert mcf.hf_subset == mcf_em.hf_subset == chat.hf_subset == subset
        assert mcf.generation_size == -1
        assert {metric.category for metric in mcf.metrics} == {SamplingMethod.LOGPROBS}
        assert mcf_em.generation_size == 5
        assert mcf_em.stop_sequence == ["\n"]
        assert {metric.category for metric in mcf_em.metrics} == {SamplingMethod.GENERATIVE}
        if subset in MCF_SUBSETS:
            assert mcf_em.prompt_function is mcf.prompt_function
        else:
            assert mcf_em.prompt_function.func is mcf.prompt_function.func
            assert mcf_em.prompt_function.keywords == mcf.prompt_function.keywords
        assert chat.generation_size == 32768
        assert chat.stop_sequence == []
        assert {metric.category for metric in chat.metrics} == {SamplingMethod.GENERATIVE}
        assert mcf.sample_fields is mcf_em.sample_fields is chat.sample_fields
        if subset in MCF_SUBSETS:
            assert mcf.sample_fields is record_to_sample


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (
            "bigbench_hard",
            {
                f"bigbench_hard:{subset}:{variant}"
                for subset in MCF_SUBSETS | BINARY_SUBSETS
                for variant in ("mcf", "mcf_em", "chat")
            },
        ),
        ("bigbench_hard:mcf", {f"bigbench_hard:{subset}:mcf" for subset in MCF_SUBSETS | BINARY_SUBSETS}),
        ("bigbench_hard:mcf_em", {f"bigbench_hard:{subset}:mcf_em" for subset in MCF_SUBSETS | BINARY_SUBSETS}),
        ("bigbench_hard:chat", {f"bigbench_hard:{subset}:chat" for subset in MCF_SUBSETS | BINARY_SUBSETS}),
        ("bigbench_hard:navigate", {f"bigbench_hard:navigate:{v}" for v in ("mcf", "mcf_em", "chat")}),
        ("bigbench_hard:navigate:mcf", {"bigbench_hard:navigate:mcf"}),
        (
            "bigbench_hard:boolean_expressions",
            {f"bigbench_hard:boolean_expressions:{v}" for v in ("mcf", "mcf_em", "chat")},
        ),
        ("bigbench_hard:formal_fallacies", {f"bigbench_hard:formal_fallacies:{v}" for v in ("mcf", "mcf_em", "chat")}),
        (
            "bigbench_hard:date_understanding",
            {f"bigbench_hard:date_understanding:{v}" for v in ("mcf", "mcf_em", "chat")},
        ),
    ],
)
def test_variant_group_selection(monkeypatch, selection, expected):
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: CONFIGS))
    registry = Registry(f"{selection}|3")
    assert set(registry.task_to_configs) == expected
    assert set(registry.load_tasks()) == {f"{name}|3" for name in expected}


@pytest.mark.parametrize("variant", ["mcf", "mcf_em", "chat"])
@pytest.mark.parametrize("subset", sorted(YES_NO_SUBSETS))
@pytest.mark.parametrize("target", ["Yes", "No", "yes", "no", " YES ", " nO "])
@pytest.mark.parametrize("native_options", ["", "\nOptions:\n- Yes\n- No"])
def test_yes_no_options_and_golds_agree_across_backends(variant, subset, target, native_options):
    config = CONFIGS[f"bigbench_hard:{subset}:{variant}"]
    record = {"input": f"Does this statement hold?{native_options}", "target": target}
    original = deepcopy(record)
    doc = config.prompt_function(record, task_name=config.name)
    sample = config.sample_fields(record)
    assert doc.task_name == config.name
    assert doc.choices == [" A", " B"]
    assert sample.input == "Does this statement hold?"
    assert set(sample.choices) == {"Yes", "No"}
    assert sample.choices[doc.gold_index].casefold() == target.strip().casefold()
    assert sample.target == ascii_uppercase[doc.gold_index]
    assert doc.get_golds() == [f" {sample.target}"]
    for label, option in zip("AB", sample.choices):
        assert f"\n{label}: {option}" in doc.query
        assert f"{label}) {option}" in sample_to_fewshot(sample)
    assert "Options:" not in doc.query
    assert sample_to_fewshot(sample).endswith(f"ANSWER: {sample.target}")
    assert record == original


@pytest.mark.parametrize("variant", ["mcf", "mcf_em", "chat"])
@pytest.mark.parametrize("subset", sorted(BINARY_SUBSETS))
def test_binary_shuffle_is_stable_independent_of_gold_and_global_random_state(variant, subset):
    config = CONFIGS[f"bigbench_hard:{subset}:{variant}"]
    choices = BINARY_CHOICES[subset]
    options = "\n".join(f"- {choice}" for choice in choices)
    record = {"input": f"Does this statement hold?\nOptions:\n{options}", "target": choices[0]}
    rng_state = random.getstate()
    doc = config.prompt_function(record)
    sample = config.sample_fields(record)
    assert random.getstate() == rng_state
    assert config.prompt_function({**record, "__index": 99, "__few_shots": True}) == doc
    assert config.sample_fields({**record, "input": "Does this statement hold?"}).choices == sample.choices
    flipped = config.prompt_function({**record, "target": choices[1]})
    assert flipped.query == doc.query
    assert flipped.gold_index == 1 - doc.gold_index
    try:
        random.seed(12345)
        changed_state = random.getstate()
        assert config.prompt_function(record) == doc
        assert random.getstate() == changed_state
    finally:
        random.setstate(rng_state)


@pytest.mark.parametrize("subset", sorted(BINARY_SUBSETS))
def test_binary_uses_both_label_orders_and_scores_the_mapped_answer(subset):
    config = CONFIGS[f"bigbench_hard:{subset}:mcf"]
    choices = BINARY_CHOICES[subset]
    first_options = set()
    for i in range(32):
        record = {"input": f"Is statement {i} plausible?", "target": choices[0]}
        doc = config.prompt_function(record)
        first_options.add(config.sample_fields(record).choices[0])
        for correct in (True, False):
            predicted_index = doc.gold_index if correct else 1 - doc.gold_index
            logprobs = [-10.0, -10.0]
            logprobs[predicted_index] = -0.1
            assert apply_metric([ModelResponse(logprobs=logprobs)], [doc], config.metrics) == [
                {"acc": int(correct), "acc_norm": int(correct)}
            ]
    assert first_options == set(choices)


@pytest.mark.parametrize("variant", ["mcf", "mcf_em", "chat"])
@pytest.mark.parametrize("subset", sorted(BINARY_SUBSETS))
def test_binary_fewshots_preserve_mapping_and_exclude_the_evaluated_row(variant, subset):
    config = replace(CONFIGS[f"bigbench_hard:{subset}:{variant}"], num_fewshots=1)
    task = LightevalTask(config)
    target = BINARY_CHOICES[subset][0]
    task.dataset = {"test": [{"input": f"Does statement {i} hold?", "target": target} for i in range(3)]}
    for doc in task.get_docs():
        assert len(doc.fewshot_samples) == 1
        assert doc.fewshot_samples[0].source_id != doc.source_id
        for candidate in [doc, *doc.fewshot_samples]:
            label = candidate.get_golds()[0].strip()
            assert f"\n{label}: {target}\n" in candidate.query


@pytest.mark.parametrize("variant", ["mcf", "mcf_em", "chat"])
@pytest.mark.parametrize("target", ["True", "invalid", "(A)"])
def test_yes_no_rejects_unexpected_answer_formats(variant, target):
    config = CONFIGS[f"bigbench_hard:causal_judgement:{variant}"]
    record = {"input": "Question?", "target": target}
    with pytest.raises(ValueError, match="Yes or No gold answer"):
        config.prompt_function(record)
    with pytest.raises(ValueError, match="Yes or No gold answer"):
        config.sample_fields(record)


@pytest.mark.parametrize(
    "subset,target", [(subset, target) for subset, choices in BINARY_CHOICES.items() for target in choices]
)
def test_binary_chat_matches_mcf_mapping_and_extracts_the_answer(subset, target):
    mcf = CONFIGS[f"bigbench_hard:{subset}:mcf"]
    chat = CONFIGS[f"bigbench_hard:{subset}:chat"]
    gold_labels = set()
    for i in range(16):
        record = {"input": f"Does statement {i} hold?", "target": target}
        mcf_doc = mcf.prompt_function(record)
        chat_doc = chat.prompt_function(record)
        assert chat_doc.choices == mcf_doc.choices
        assert chat_doc.gold_index == mcf_doc.gold_index
        assert chat.sample_fields(record) == mcf.sample_fields(record)
        assert "Think step by step before answering." in chat_doc.instruction
        assert "LETTER is one of AB." in chat_doc.instruction
        for line in mcf_doc.query.splitlines():
            if line.startswith(("A:", "B:")):
                assert line in chat_doc.query.splitlines()
        gold_labels.add(chat_doc.get_golds()[0].strip())
        for correct in (True, False):
            index = chat_doc.gold_index if correct else 1 - chat_doc.gold_index
            answer = ascii_uppercase[index]
            response = ModelResponse(text=[f"After reasoning through the question.\nAnswer: {answer}"])
            assert apply_metric([response], [chat_doc], chat.metrics) == [{"extractive_match": float(correct)}]
    assert gold_labels == {"A", "B"}


@pytest.mark.parametrize("variant", ["mcf", "mcf_em", "chat"])
@pytest.mark.parametrize("subset", ["boolean_expressions", "formal_fallacies"])
@pytest.mark.parametrize("answer_index", [0, 1])
@pytest.mark.parametrize("native_options", [False, True])
def test_new_binary_options_and_golds_agree_across_backends(variant, subset, answer_index, native_options):
    config = CONFIGS[f"bigbench_hard:{subset}:{variant}"]
    choices = BINARY_CHOICES[subset]
    question = "Does this statement hold?"
    if native_options:
        question += "\nOptions:\n" + "\n".join(f"- {choice} " for choice in choices)
    answer = choices[answer_index]
    for target in (answer, answer.upper(), answer.lower(), f" {answer} "):
        record = {"input": question, "target": target}
        original = deepcopy(record)
        doc = config.prompt_function(record, task_name=config.name)
        sample = config.sample_fields(record)
        assert doc.task_name == config.name
        assert doc.choices == [" A", " B"]
        assert sample.input == "Does this statement hold?"
        assert set(sample.choices) == set(choices)
        assert sample.choices[doc.gold_index] == answer
        assert sample.target == ascii_uppercase[doc.gold_index]
        assert doc.get_golds() == [f" {sample.target}"]
        for label, option in zip("AB", sample.choices):
            assert f"\n{label}: {option}" in doc.query
            assert f"{label}) {option}" in sample_to_fewshot(sample)
        assert "Options:" not in doc.query
        assert sample_to_fewshot(sample).endswith(f"ANSWER: {sample.target}")
        assert record == original


@pytest.mark.parametrize("variant", ["mcf", "mcf_em", "chat"])
@pytest.mark.parametrize(
    "subset,target",
    [
        ("boolean_expressions", "Yes"),
        ("boolean_expressions", "valid"),
        ("formal_fallacies", "True"),
        ("formal_fallacies", "No"),
    ],
)
def test_new_binary_subtasks_reject_answers_from_other_families(variant, subset, target):
    config = CONFIGS[f"bigbench_hard:{subset}:{variant}"]
    record = {"input": "Question?", "target": target}
    with pytest.raises(ValueError, match="binary records must have"):
        config.prompt_function(record)
    with pytest.raises(ValueError, match="binary records must have"):
        config.sample_fields(record)
