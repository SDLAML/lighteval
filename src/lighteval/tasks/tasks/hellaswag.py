"""
name:
Hellaswag

dataset:
Rowan/hellaswag

abstract:
HellaSwag is a commonsense inference benchmark designed to challenge language
models with adversarially filtered multiple-choice questions.

languages:
english

tags:
multiple-choice, narrative, reasoning

paper:
https://arxiv.org/abs/1905.07830
"""

from string import ascii_uppercase
import re

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.metrics.metrics import Metrics
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


def harness_preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def hellaswag_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options in prompt, score label tokens via logprobs."""
    ctx = line["ctx_a"] + " " + line["ctx_b"].capitalize()
    query = harness_preprocess(line["activity_label"] + ": " + ctx)
    query += "".join(
        [f"\n{key}. {harness_preprocess(choice)}" for key, choice in zip(ascii_uppercase, line["endings"])]
    )
    query += "\nAnswer:"

    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in ascii_uppercase[: len(line["endings"])]],
        gold_index=gold_ix,
    )

def hellaswag_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt, score full answer texts via logprobs."""
    ctx = line["ctx_a"] + " " + line["ctx_b"].capitalize()
    query = harness_preprocess(line["activity_label"] + ": " + ctx)
    choices = [harness_preprocess(ending) for ending in line["endings"]]
    choices = [c if c and c[0].isspace() else " " + c for c in choices]
    gold_ix = int(line["label"]) if str(line.get("label", "")).strip() != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_ix,
    )


# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
hellaswag_mcf = LightevalTaskConfig(
    name="hellaswag:mcf",
    prompt_function=hellaswag_mcf_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# Greedy variant: MCF-style prompt, generate 1 token, exact match
hellaswag_mcf_em = LightevalTaskConfig(
    name="hellaswag:mcf_em",
    prompt_function=hellaswag_mcf_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)
# CF variant: completion-style, logprob on full answer text + BPB on gold choice
hellaswag_cf = LightevalTaskConfig(
    name="hellaswag:cf",
    prompt_function=hellaswag_cf_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    hellaswag_mcf,
    hellaswag_mcf_em,
    hellaswag_cf,
]
