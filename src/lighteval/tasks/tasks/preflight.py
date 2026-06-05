"""
name:
Preflight Aviation Safety QA

dataset:
titaneval_local (local parquet — data/titaneval/preflight.parquet)

abstract:
Aviation safety multiple-choice questions derived from international airport ground
operations manuals and FAA/ICAO regulations. 300 questions, 0-shot.

languages:
english

tags:
aviation, multiple-choice, qa, safety
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


def preflight_cf_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in line["choices"]],
        gold_index=line["answer_index"],
    )


def preflight_mcf_prompt(line, task_name: str = None):
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
        name=f"preflight:{suffix}",
        prompt_function=fn,
        hf_repo="titaneval_local",
        hf_subset="preflight",
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
        ("cf",     preflight_cf_prompt,  _CF_METRICS,          -1),
        ("mcf",    preflight_mcf_prompt, _MCF_METRICS,         -1),
        ("mcf_em", preflight_mcf_prompt, [Metrics.exact_match],  1),
    ]
]
