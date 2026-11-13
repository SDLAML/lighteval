"""
name:
ChemBench Chemistry QA

dataset:
titaneval_local (local parquet — data/titaneval/chembench.parquet)

abstract:
Chemistry multiple-choice benchmark covering analytical, organic, and physical
chemistry. 2,542 questions from TitanEval-MCQ (ChemBench, jablonkagroup/ChemBench,
multiple-choice part). Version 1 (owner 2026-11-13): the 1,000 `chemical_preference`
items (domain `chemistry_chemical_preference`) are dropped, 1,542 are scored: they are
2-option taste questions ("which of two molecules would you prefer") with no checkable
answer, and a base model answers "A" almost always (accuracy 0.970 when A is gold, 0.059
when B is), so they add position noise to 39% of the weight. The same filtered split is
the few-shot pool, so no preference item is a demonstration either. Dropped by
`drop_reason`; the count is in configs/tasks/lighteval/chembench_mcf_em.yaml (notes).

languages:
english

tags:
chemistry, multiple-choice, qa, science
"""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.domain_drops import KeepRows

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]
_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def chembench_cf_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in line["choices"]],
        gold_index=line["answer_index"],
    )


def chembench_mcf_prompt(line, task_name: str = None):
    choices = line["choices"]
    labels = list("ABCDEFGHIJ"[: len(choices)])
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=line["answer_index"],
    )


DROPPED_DOMAIN = "chemistry_chemical_preference"        # 1,000 rows of 2,542


def drop_reason(row):
    """None = scored; "chemical_preference" = a taste question without a checkable answer (field `domain`)."""
    return "chemical_preference" if row.get("domain") == DROPPED_DOMAIN else None


# The digest of the dropped value is part of the datasets cache key (see domain_drops); bump `version` and `rule` with every change of the rule.
_KEEP = KeepRows("chembench-v1: drop domain chemistry_chemical_preference", drop_reason, parts=[DROPPED_DOMAIN])


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"chembench:{suffix}",
        prompt_function=fn,
        hf_repo="titaneval_local",
        hf_subset="chembench",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=gen,
        metrics=metrics,
        hf_filter=_KEEP,
        stop_sequence=["\n"],
        version=1,
    )
    for suffix, fn, metrics, gen in [
        ("cf",     chembench_cf_prompt,  _CF_METRICS,          -1),
        ("mcf",    chembench_mcf_prompt, _MCF_METRICS,         -1),
        ("mcf_em", chembench_mcf_prompt, [Metrics.exact_match],  1),
    ]
]
