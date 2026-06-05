"""
name:
FormationEval

dataset:
AlmazErmilov/FormationEval

abstract:
FormationEval is a multiple-choice benchmark for formation evaluation and
petroleum engineering knowledge.

languages:
english

tags:
multiple-choice, petroleum, qa

paper:
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

_LABELS = list("ABCDE")


def formationeval_cf_prompt(line, task_name: str = None):
    import ast
    choices = ast.literal_eval(line["choices"]) if isinstance(line["choices"], str) else line["choices"]
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=int(line["answer_index"]),
    )


def formationeval_mcf_prompt(line, task_name: str = None):
    import ast
    choices = ast.literal_eval(line["choices"]) if isinstance(line["choices"], str) else line["choices"]
    labels = _LABELS[: len(choices)]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=int(line["answer_index"]),
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"formationeval:{suffix}",
        prompt_function=fn,
        hf_repo="AlmazErmilov/FormationEval",
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
        ("cf",     formationeval_cf_prompt,  _CF_METRICS,             -1),
        ("mcf",    formationeval_mcf_prompt, _MCF_METRICS,            -1),
        ("mcf_em", formationeval_mcf_prompt, [Metrics.exact_match],    1),
    ]
]
