"""
name:
Boolq

dataset:
lighteval/boolq_helm

abstract:
The BoolQ benchmark for binary (yes/no) question answering.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1905.11946
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


def _question(line) -> str:
    q = line["question"]
    return q[:-1] if q.endswith("??") else q


def _gold(line) -> int:
    return 0 if line["answer"] == "Yes" else 1


def boolq_cf_prompt(line, task_name: str = None):
    """CF variant: score full answer text via logprobs."""
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {_question(line)}\nAnswer:",
        choices=[" Yes", " No"],
        gold_index=_gold(line),
    )


def boolq_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B options, score label tokens via logprobs."""
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {_question(line)}\n A. Yes\n B. No\nAnswer:",
        choices=[" A", " B"],
        gold_index=_gold(line),
    )


# lighteval/boolq_helm: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "fdd8158ec36960949eb9ca6285f3c93156c07d5d"


TASKS_TABLE = [
    LightevalTaskConfig(
        name="boolq:cf",
        prompt_function=boolq_cf_prompt,
        hf_repo="lighteval/boolq_helm",
        hf_revision=_REVISION,
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=_CF_METRICS,
        stop_sequence=["\n"],
        version=1,
    ),
    LightevalTaskConfig(
        name="boolq:mcf",
        prompt_function=boolq_mcf_prompt,
        hf_repo="lighteval/boolq_helm",
        hf_revision=_REVISION,
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=1,
    ),
    LightevalTaskConfig(
        name="boolq:mcf_em",
        prompt_function=boolq_mcf_prompt,
        hf_repo="lighteval/boolq_helm",
        hf_revision=_REVISION,
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=1,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=1,
    ),
]
