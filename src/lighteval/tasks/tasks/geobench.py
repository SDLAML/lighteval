"""
name:
GeoBench Geoscience QA

dataset:
titaneval_local (local parquet — data/titaneval/geobench.parquet)

abstract:
Geoscience multiple-choice benchmark. 1,390 questions from TitanEval-MCQ
(originally GeoBench-VLM, text-only subset used here), 0-shot.

languages:
english

tags:
geoscience, multiple-choice, qa, science
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


def geobench_cf_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in line["choices"]],
        gold_index=line["answer_index"],
    )


def geobench_mcf_prompt(line, task_name: str = None):
    choices = line["choices"]
    labels = list("ABCDEFGHIJ"[: len(choices)])
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=line["answer_index"],
    )



TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"geobench:{suffix}",
        prompt_function=fn,
        hf_repo="titaneval_local",
        hf_subset="geobench",
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
        ("cf",     geobench_cf_prompt,  _CF_METRICS,          -1),
        ("mcf",    geobench_mcf_prompt, _MCF_METRICS,         -1),
        ("mcf_em", geobench_mcf_prompt, [Metrics.exact_match],  1),
    ]
]
