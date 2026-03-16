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
    letters = list(ascii_uppercase)[:2]
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['goal']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(letters, [line["sol1"], line["sol2"]])])
    query += "Answer:"

    gold_ix = int(line["label"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + l for l in letters],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


def piqa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt, score full answer texts via logprobs."""
    query = f"Question: {line['goal']}\nAnswer:"
    choices = [line["sol1"], line["sol2"]]
    choices = [c if c and c[0].isspace() else " " + c for c in choices]
    gold_ix = int(line["label"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_ix,
    )


def piqa_bpb_prompt(line, task_name: str = None):
    """BPB variant: CF-style prompt with only the gold solution."""
    gold_ix = int(line["label"])
    gold_text = line["sol1"] if gold_ix == 0 else line["sol2"]
    if not gold_text:
        return None
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Question: {line['goal']}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )


# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
piqa_mcf = LightevalTaskConfig(
    name="piqa:mcf",
    prompt_function=piqa_mcf_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
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
piqa_mcf_em = LightevalTaskConfig(
    name="piqa:mcf_em",
    prompt_function=piqa_mcf_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
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
    few_shots_split=None,
    few_shots_select=None,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    piqa_mcf,
    piqa_mcf_em,
    piqa_cf,
]
