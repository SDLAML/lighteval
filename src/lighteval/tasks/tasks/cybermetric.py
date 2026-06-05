"""
name:
CyberMetric + SecQA

datasets:
tihanyin/CyberMetric
zefang-liu/secqa (secqa_v1, secqa_v2)

abstract:
Cybersecurity multiple-choice benchmarks. CyberMetric covers cybersecurity
concepts. SecQA (Security QA) consists of expert-written 4-choice security
questions in two versions of increasing difficulty.

languages:
english

tags:
cybersecurity, multiple-choice, qa

paper:
https://arxiv.org/abs/2411.02228 (CyberMetric)
"""

import ast

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


# ---- CyberMetric ----

def _cybermetric_parse(line):
    """Parse nested 'questions' field into question/choices/gold."""
    d = line["questions"]
    if isinstance(d, str):
        d = ast.literal_eval(d)
    keys = sorted(d["answers"].keys())
    choices = [d["answers"][k] for k in keys]
    answer_key = d.get("correct_solution") or d.get("solution")
    gold = keys.index(answer_key)
    return d["question"], choices, gold


def cybermetric_cf_prompt(line, task_name: str = None):
    question, choices, gold = _cybermetric_parse(line)
    return Doc(
        task_name=task_name,
        query=f"Question: {question}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=gold,
    )


def cybermetric_mcf_prompt(line, task_name: str = None):
    question, choices, gold = _cybermetric_parse(line)
    labels = list("ABCD")[: len(choices)]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {question}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=gold,
    )


# ---- SecQA ----

def secqa_cf_prompt(line, task_name: str = None):
    choices = [line["A"], line["B"], line["C"], line["D"]]
    return Doc(
        task_name=task_name,
        query=f"Question: {line['Question']}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=list("ABCD").index(line["Answer"]),
    )


def secqa_mcf_prompt(line, task_name: str = None):
    choices = [line["A"], line["B"], line["C"], line["D"]]
    options = "\n".join(f" {l}. {c}" for l, c in zip("ABCD", choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['Question']}\n{options}\nAnswer:",
        choices=[" A", " B", " C", " D"],
        gold_index=list("ABCD").index(line["Answer"]),
    )


def _cybermetric_configs():
    return [
        LightevalTaskConfig(
            name=f"cybermetric:{suffix}",
            prompt_function=fn,
            hf_repo="tihanyin/CyberMetric",
            hf_subset="default",
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
            ("cf",     cybermetric_cf_prompt,  _CF_METRICS,             -1),
            ("mcf",    cybermetric_mcf_prompt, _MCF_METRICS,            -1),
            ("mcf_em", cybermetric_mcf_prompt, [Metrics.exact_match],    1),
        ]
    ]


def _secqa_configs(version: str):
    return [
        LightevalTaskConfig(
            name=f"secqa:{version}:{suffix}",
            prompt_function=fn,
            hf_repo="zefang-liu/secqa",
            hf_subset=f"secqa_{version}",
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
            ("cf",     secqa_cf_prompt,  _CF_METRICS,             -1),
            ("mcf",    secqa_mcf_prompt, _MCF_METRICS,            -1),
            ("mcf_em", secqa_mcf_prompt, [Metrics.exact_match],    1),
        ]
    ]


TASKS_TABLE = (
    _cybermetric_configs()
    + _secqa_configs("v1")
    + _secqa_configs("v2")
)
