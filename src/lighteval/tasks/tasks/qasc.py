"""
name:
Qasc

dataset:
allenai/qasc

abstract:
QASC is a question-and-answer dataset that focuses on sentence composition.
It consists of 8-way multiple choice questions requiring combining two facts
from a large corpus to derive an answer.

languages:
english

tags:
multiple-choice, qa, reasoning, science

paper:
https://arxiv.org/abs/1910.11473
"""

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

_LABELS = list("ABCDEFGH")


def qasc_cf_prompt(line, task_name: str = None):
    """CF variant: score full answer texts via logprobs."""
    choices = line["choices"]["text"]
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=_LABELS.index(line["answerKey"]),
    )


def qasc_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A-H options, score label tokens via logprobs."""
    choices = line["choices"]["text"]
    labels = _LABELS[: len(choices)]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=_LABELS.index(line["answerKey"]),
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="qasc:cf",
        prompt_function=qasc_cf_prompt,
        hf_repo="allenai/qasc",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=_CF_METRICS,
        stop_sequence=["\n"],
        version=0,
    ),
    LightevalTaskConfig(
        name="qasc:mcf",
        prompt_function=qasc_mcf_prompt,
        hf_repo="allenai/qasc",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=0,
    ),
    LightevalTaskConfig(
        name="qasc:mcf_em",
        prompt_function=qasc_mcf_prompt,
        hf_repo="allenai/qasc",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=1,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=0,
    ),
]
