"""
name:
LAB-Bench (TableQA)

dataset:
futurehouse/LAB-Bench (TableQA config)

abstract:
LAB-Bench is a biology laboratory capabilities benchmark. The TableQA subset
tests reading comprehension of scientific tables with 4-choice MCQ. Note: the
original tables are provided as images; this text-only variant evaluates
without visual context.

languages:
english

tags:
biology, multiple-choice, qa, science

paper:
https://arxiv.org/abs/2407.10362
"""

import ast
import hashlib
import random

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


def _stable_choices(line):
    """Deterministic shuffle of [ideal] + distractors to avoid position bias."""
    ideal = line["ideal"]
    distractors = ast.literal_eval(line["distractors"]) if isinstance(line["distractors"], str) else line["distractors"]
    items = [ideal] + list(distractors)
    seed = int(hashlib.md5(line["question"].encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items, items.index(ideal)


def labbench_cf_prompt(line, task_name: str = None):
    choices, gold = _stable_choices(line)
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=gold,
    )


def labbench_mcf_prompt(line, task_name: str = None):
    choices, gold = _stable_choices(line)
    labels = list("ABCD")[: len(choices)]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=gold,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"labbench:{suffix}",
        prompt_function=fn,
        hf_repo="futurehouse/LAB-Bench",
        hf_subset="TableQA",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=gen,
        metrics=metrics,
        stop_sequence=["\n"],
        version=0,
    )
    for suffix, fn, metrics, gen in [
        ("cf",     labbench_cf_prompt,  _CF_METRICS,             -1),
        ("mcf",    labbench_mcf_prompt, _MCF_METRICS,            -1),
        ("mcf_em", labbench_mcf_prompt, [Metrics.exact_match],    1),
    ]
]
