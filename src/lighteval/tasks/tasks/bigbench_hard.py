"""
name:
Bigbench Hard

dataset:
lukaemon/bbh

abstract:
BIG-Bench Hard (BBH) spans 27 dataset configurations requiring multi-step
reasoning. This evaluation includes 23 multiple-choice subtasks in :mcf (label
log-likelihoods), :mcf_em (generated label exact match), and :chat (reasoning
followed by a final answer letter) variants.
The six binary subtasks (Yes/No, True/False, valid/invalid) use reproducibly
randomized A/B options. The four open-ended subtasks are excluded.

languages:
english

tags:
reasoning

paper:
https://arxiv.org/abs/2210.09261
"""

import random
import re
from functools import partial
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Excluded from this multiple-choice evaluation because their answers are open-ended:
# dyck_languages (bracket sequences), multistep_arithmetic_two (integers),
# object_counting (counts), and word_sorting (sorted word lists).
_BBH_MCF_SUBSETS = [
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
]

_BBH_BINARY_CHOICES = {
    "boolean_expressions": ("True", "False"),
    "causal_judgement": ("Yes", "No"),
    "formal_fallacies": ("valid", "invalid"),
    "navigate": ("Yes", "No"),
    "sports_understanding": ("Yes", "No"),
    "web_of_lies": ("Yes", "No"),
}

_BBH_OPTIONS_RE = re.compile(r"^\(([A-Z])\)[ \t]*(.*?)(?=^\([A-Z]\)|\Z)", re.MULTILINE | re.DOTALL)


def _parse_multiple_choice(line):
    """Parse letter-labeled options, skipping records with a free-text gold."""
    target = re.fullmatch(r"\(([A-Z])\)", line["target"].strip())
    if target is None:
        # Three canonical movie_recommendation/ruin_names records have broken
        # options and free-text targets that cannot be mapped to a single label.
        return None

    question, separator, option_text = line["input"].partition("\nOptions:\n")
    options = _BBH_OPTIONS_RE.findall(option_text)
    labels = "".join(label for label, _ in options)
    choices = [text.strip() for _, text in options]
    if (
        not separator
        or not choices
        or labels != ascii_uppercase[: len(choices)]
        or not all(choices)
        or target[1] not in labels
    ):
        raise ValueError("BBH multiple-choice records must have consecutive labels containing the gold answer.")
    return question.strip(), choices, labels.index(target[1])


def _parse_binary(line, *, choices):
    """Randomize binary option order reproducibly using only the question text."""
    question = line["input"].partition("\nOptions:\n")[0].strip()
    target = line["target"].strip().casefold()
    if target not in {choice.casefold() for choice in choices}:
        raise ValueError(f"BBH binary records must have a {' or '.join(choices)} gold answer.")

    choices = list(choices)
    # A local, question-seeded RNG keeps evaluation, few-shots, and Inspect in
    # agreement, independently of gold answers, row order, and global RNG state.
    random.Random(question).shuffle(choices)
    gold_index = [choice.casefold() for choice in choices].index(target)
    return question, choices, gold_index


def bbh_mcf_prompt(line, task_name: str = None, *, parse_record=_parse_multiple_choice):
    """Show all options and score their letter labels, as in SuperGPQA."""
    parsed = parse_record(line)
    if parsed is None:
        return None
    question, choices, gold_index = parsed
    labels = ascii_uppercase[: len(choices)]
    options = "\n".join(f"{label}: {option}" for label, option in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Answer the following multiple choice question.\n\nQuestion: {question}\n{options}\nAnswer:",
        choices=[f" {label}" for label in labels],
        gold_index=gold_index,
    )


def bbh_chat_prompt(line, task_name: str = None, *, parse_record=_parse_multiple_choice):
    """Request reasoning followed by a final answer letter, as in SuperGPQA CoT."""
    parsed = parse_record(line)
    if parsed is None:
        return None
    question, choices, gold_index = parsed
    labels = ascii_uppercase[: len(choices)]
    instruction = (
        "Answer the following multiple choice question. The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
        f"{labels}. Think step by step before answering."
    )
    options = "\n".join(f"{label}: {option}" for label, option in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"{instruction}\n\n{question}\n\n{options}\n\nAnswer:",
        choices=[f" {label}" for label in labels],
        gold_index=gold_index,
        instruction=instruction,
    )


def record_to_sample(record, *, parse_record=_parse_multiple_choice):
    parsed = parse_record(record)
    if parsed is None:
        return []
    question, choices, gold_index = parsed
    return Sample(input=question, choices=choices, target=ascii_uppercase[gold_index])


def sample_to_fewshot(sample):
    options = "\n".join(f"{label}) {option}" for label, option in zip(ascii_uppercase, sample.choices))
    return f"{sample.input}\n\n{options}\n\nANSWER: {sample.target}"


_BBH_MCF_COMMON = {
    "hf_repo": "lukaemon/bbh",
    "hf_avail_splits": ["test"],
    "evaluation_splits": ["test"],
    "few_shots_split": "test",
    "few_shots_select": "random_sampling",
    "sample_fields": record_to_sample,
    "sample_to_fewshot": sample_to_fewshot,
    "scorer": choice(),
    "version": 0,
}

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"bigbench_hard:{subset}:mcf",
        prompt_function=bbh_mcf_prompt,
        hf_subset=subset,
        solver=[multiple_choice(cache=True)],
        generation_size=-1,
        metrics=[LogLikelihoodAccMetric(), LogLikelihoodAccMetric(normalization=LogProbCharNorm())],
        stop_sequence=["\n"],
        **_BBH_MCF_COMMON,
    )
    for subset in _BBH_MCF_SUBSETS
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"bigbench_hard:{subset}:mcf_em",
        prompt_function=bbh_mcf_prompt,
        hf_subset=subset,
        solver=[multiple_choice(cache=True)],
        generation_size=5,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        **_BBH_MCF_COMMON,
    )
    for subset in _BBH_MCF_SUBSETS
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"bigbench_hard:{subset}:chat",
        prompt_function=bbh_chat_prompt,
        hf_subset=subset,
        solver=[multiple_choice(cot=True, cache=True)],
        generation_size=32768,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=[],
        **_BBH_MCF_COMMON,
    )
    for subset in _BBH_MCF_SUBSETS
]


_BBH_BINARY_PARSERS = {
    subset: partial(_parse_binary, choices=choices) for subset, choices in _BBH_BINARY_CHOICES.items()
}
_BBH_BINARY_COMMON = {
    subset: {**_BBH_MCF_COMMON, "sample_fields": partial(record_to_sample, parse_record=parser)}
    for subset, parser in _BBH_BINARY_PARSERS.items()
}

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"bigbench_hard:{subset}:mcf",
        prompt_function=partial(bbh_mcf_prompt, parse_record=parser),
        hf_subset=subset,
        solver=[multiple_choice(cache=True)],
        generation_size=-1,
        metrics=[LogLikelihoodAccMetric(), LogLikelihoodAccMetric(normalization=LogProbCharNorm())],
        stop_sequence=["\n"],
        **_BBH_BINARY_COMMON[subset],
    )
    for subset, parser in _BBH_BINARY_PARSERS.items()
]


TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"bigbench_hard:{subset}:mcf_em",
        prompt_function=partial(bbh_mcf_prompt, parse_record=parser),
        hf_subset=subset,
        solver=[multiple_choice(cache=True)],
        generation_size=5,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        **_BBH_BINARY_COMMON[subset],
    )
    for subset, parser in _BBH_BINARY_PARSERS.items()
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"bigbench_hard:{subset}:chat",
        prompt_function=partial(bbh_chat_prompt, parse_record=parser),
        hf_subset=subset,
        solver=[multiple_choice(cot=True, cache=True)],
        generation_size=32768,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=[],
        **_BBH_BINARY_COMMON[subset],
    )
    for subset, parser in _BBH_BINARY_PARSERS.items()
]
