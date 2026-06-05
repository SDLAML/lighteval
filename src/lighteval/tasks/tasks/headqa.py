"""
name:
Headqa

dataset:
lighteval/headqa_harness

abstract:
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to
access a specialized position in the Spanish healthcare system, and are
challenging even for highly specialized humans. They are designed by the
Ministerio de Sanidad, Consumo y Bienestar Social, who also provides direct
access to the exams of the last 5 years.

languages:
english, spanish

tags:
health, medical, multiple-choice, qa

paper:
https://arxiv.org/abs/1906.04701
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


def headqa_cf_prompt(line, task_name: str = None):
    """CF variant: score full answer texts via logprobs."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['qtext']}\nAnswer:",
        choices=[f" {answer['atext']}" for answer in line["answers"]],
        gold_index=int(line["ra"]) - 1,
    )


def headqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options, score label tokens via logprobs."""
    labels = list("ABCDE")[: len(line["answers"])]
    options = "\n".join(f" {l}. {a['atext']}" for l, a in zip(labels, line["answers"]))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['qtext']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=int(line["ra"]) - 1,
    )


def _configs(lang: str):
    return [
        LightevalTaskConfig(
            name=f"headqa:{lang}:cf",
            prompt_function=headqa_cf_prompt,
            hf_repo="lighteval/headqa_harness",
            hf_subset=lang,
            hf_avail_splits=["train", "test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling_from_train",
            generation_size=-1,
            metrics=_CF_METRICS,
            stop_sequence=["\n"],
            version=1,
        ),
        LightevalTaskConfig(
            name=f"headqa:{lang}:mcf",
            prompt_function=headqa_mcf_prompt,
            hf_repo="lighteval/headqa_harness",
            hf_subset=lang,
            hf_avail_splits=["train", "test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling_from_train",
            generation_size=-1,
            metrics=_MCF_METRICS,
            stop_sequence=["\n"],
            version=1,
        ),
        LightevalTaskConfig(
            name=f"headqa:{lang}:mcf_em",
            prompt_function=headqa_mcf_prompt,
            hf_repo="lighteval/headqa_harness",
            hf_subset=lang,
            hf_avail_splits=["train", "test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling_from_train",
            generation_size=1,
            metrics=[Metrics.exact_match],
            stop_sequence=["\n"],
            version=1,
        ),
    ]


TASKS_TABLE = _configs("en") + _configs("es")
