"""
name:
XFinBench Cross-lingual Finance QA

dataset:
titaneval_local (local parquet — data/titaneval/xfinbench.parquet)

abstract:
Cross-lingual finance multiple-choice benchmark (English subset). 588 valid
questions from TitanEval-MCQ, 0-shot.

languages:
english

tags:
finance, multiple-choice, qa
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


def xfinbench_cf_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in line["choices"]],
        gold_index=line["answer_index"],
    )


def xfinbench_mcf_prompt(line, task_name: str = None):
    choices = line["choices"]
    labels = list("ABCDEFGHIJ"[: len(choices)])
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=line["answer_index"],
    )


def _valid_row(row) -> bool:
    """Skip rows with empty choices (xfinbench has a few invalid rows)."""
    choices = row.get("choices") or []
    ans = row.get("answer_index")
    return bool(choices) and any(c.strip() for c in choices) and ans is not None and 0 <= ans < len(choices)



TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"xfinbench:{suffix}",
        prompt_function=fn,
        hf_repo="titaneval_local",
        hf_subset="xfinbench",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=gen,
        metrics=metrics,
        hf_filter=_valid_row,
        stop_sequence=["\n"],
        version=0,
    )
    for suffix, fn, metrics, gen in [
        ("cf",     xfinbench_cf_prompt,  _CF_METRICS,          -1),
        ("mcf",    xfinbench_mcf_prompt, _MCF_METRICS,         -1),
        ("mcf_em", xfinbench_mcf_prompt, [Metrics.exact_match],  1),
    ]
]
