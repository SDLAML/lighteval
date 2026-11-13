"""
name:
MaScQA

dataset:
heegyu/mascqa

abstract:
MaScQA (Materials Science Question Answering) is a multiple-choice benchmark
for materials science knowledge, sourced from standardized exams. Answer choices
are embedded in the question text in (A) / (B) / (C) / (D) format.

languages:
english

tags:
materials-science, multiple-choice, qa, science

paper:
https://arxiv.org/abs/2209.09088
"""

import re

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]

_LABELS = list("ABCDE")


def _parse(line):
    """Extract question, choices and labels from embedded format '... (A) c1 (B) c2 ...'."""
    text = line["questions"].strip()
    m = re.search(r'\s*\(A\)', text)
    question = text[: m.start()].strip() if m else text
    raw = re.findall(r'\(([A-E])\)\s*(.*?)(?=\s*\([A-E]\)|$)', text)
    choices = [v.strip() for _, v in raw]
    labels = [k for k, _ in raw]
    return question, choices, labels


def _has_gold(line):
    """Keep only rows whose gold is determinable: options exist and the label names one of them.

    Rows with numeric answers (no options in the text) or with a label that matches no option
    (e.g. 'B; D') are dropped; their gold cannot be determined and must not default to A.
    """
    _, choices, labels = _parse(line)
    # labels must be A, B, C, ... in order: stray '(X)' tokens inside the question text
    # produce duplicate or out-of-order labels, and labels.index would then pick a wrong option.
    return bool(choices) and labels == _LABELS[: len(labels)] and line["label"] in labels


def _gold(line, labels):
    # Only called on rows that passed _has_gold; labels.index raises ValueError otherwise.
    return labels.index(line["label"])


def mascqa_cf_prompt(line, task_name: str = None):
    question, choices, labels = _parse(line)
    return Doc(
        task_name=task_name,
        query=f"Question: {question}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=_gold(line, labels),
    )


def mascqa_mcf_prompt(line, task_name: str = None):
    question, choices, labels = _parse(line)
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {question}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=_gold(line, labels),
    )


# heegyu/mascqa: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "7f824b750da5b09ac7fd8b7be10fc0fe3dec852c"


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mascqa:{suffix}",
        prompt_function=fn,
        hf_repo="heegyu/mascqa",
        hf_revision=_REVISION,
        hf_filter=_has_gold,
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=gen,
        metrics=metrics,
        stop_sequence=["\n"],
        version=0,
    )
    for suffix, fn, metrics, gen in [
        ("cf",     mascqa_cf_prompt,  _CF_METRICS,             -1),
        ("mcf",    mascqa_mcf_prompt, _MCF_METRICS,            -1),
        ("mcf_em", mascqa_mcf_prompt, [Metrics.exact_match],    1),
    ]
]
