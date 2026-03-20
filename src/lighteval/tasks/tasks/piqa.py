"""
name:
Piqa

dataset:
ybisk/piqa

abstract:
PIQA is a benchmark for testing physical commonsense reasoning. It contains
questions requiring this kind of physical commonsense reasoning.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
https://arxiv.org/abs/1911.11641
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
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


def piqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B options in prompt, score label tokens via logprobs."""
    choices = [line["sol1"], line["sol2"]]
    options = "\n".join(f" {l}. {c}" for l, c in zip(["A", "B"], choices))
    query = f"Goal: {line['goal']}\n{options}\nAnswer:"
    gold_ix = int(line["label"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B"],
        gold_index=gold_ix,
    )


def piqa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt, score full answer texts via logprobs."""
    query = f"Goal: {line['goal']}\nAnswer:"
    choices = [line["sol1"], line["sol2"]]
    choices = [c if c and c[0].isspace() else " " + c for c in choices]
    gold_ix = int(line["label"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_ix,
    )


# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
piqa_mcf = LightevalTaskConfig(
    name="piqa:mcf",
    prompt_function=piqa_mcf_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# Greedy variant: MCF-style prompt, generate 1 token, exact match
piqa_mcf_em = LightevalTaskConfig(
    name="piqa:mcf_em",
    prompt_function=piqa_mcf_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

# CF variant: completion-style, logprob on full answer text + BPB on gold choice
piqa_cf = LightevalTaskConfig(
    name="piqa:cf",
    prompt_function=piqa_cf_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    piqa_mcf,
    piqa_mcf_em,
    piqa_cf,
]
