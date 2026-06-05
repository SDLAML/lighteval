"""
name:
TeleQnA

dataset:
netop/TeleQnA (gated — must be pre-cached)

abstract:
TeleQnA is a multiple-choice benchmark covering telecommunications standards
from 3GPP, IEEE, and other telecom bodies.

languages:
english

tags:
multiple-choice, qa, telecom

paper:
https://arxiv.org/abs/2310.15051
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


def _choices(line):
    choices = line["choices"]
    if isinstance(choices, str):
        import ast
        choices = ast.literal_eval(choices)
    return choices


def teleqna_cf_prompt(line, task_name: str = None):
    choices = _choices(line)
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=int(line["answer"]),
    )


def teleqna_mcf_prompt(line, task_name: str = None):
    choices = _choices(line)
    labels = list("ABCDE")[: len(choices)]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=int(line["answer"]),
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"teleqna:{suffix}",
        prompt_function=fn,
        hf_repo="netop/TeleQnA",
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
        ("cf",     teleqna_cf_prompt,  _CF_METRICS,             -1),
        ("mcf",    teleqna_mcf_prompt, _MCF_METRICS,            -1),
        ("mcf_em", teleqna_mcf_prompt, [Metrics.exact_match],    1),
    ]
]
