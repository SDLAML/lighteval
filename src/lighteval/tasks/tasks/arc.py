"""
name:
Arc

dataset:
allenai/ai2_arc

abstract:
7,787 genuine grade-school level, multiple-choice science questions, assembled
to encourage research in advanced question-answering. The dataset is partitioned
into a Challenge Set and an Easy Set, where the former contains only questions
answered incorrectly by both a retrieval-based algorithm and a word
co-occurrence algorithm

languages:
english

tags:
multiple-choice

paper:
https://arxiv.org/abs/1803.05457
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


def arc_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )

def arc_mcf_prompt(line, task_name: str = None):
    query = f"Question: {line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["choices"]["text"])])
    query += "Answer:"

    gold_ix = line["choices"]["label"].index(line["answerKey"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in ascii_uppercase[: len(line["choices"]["text"])]],
        gold_index=gold_ix,
    )


# CF variant: completion-style, logprob on full answer text + BPB on gold choice
arc_challenge_cf = LightevalTaskConfig(
    name="arc:challenge:cf",
    prompt_function=arc_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

arc_easy_cf = LightevalTaskConfig(
    name="arc:easy:cf",
    prompt_function=arc_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
arc_challenge_mcf = LightevalTaskConfig(
    name="arc:challenge:mcf",
    prompt_function=arc_mcf_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

arc_easy_mcf = LightevalTaskConfig(
    name="arc:easy:mcf",
    prompt_function=arc_mcf_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# Greedy variant: MCF-style prompt, generate 1 token, exact match
arc_challenge_mcf_em = LightevalTaskConfig(
    name="arc:challenge:mcf_em",
    prompt_function=arc_mcf_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arc_easy_mcf_em = LightevalTaskConfig(
    name="arc:easy:mcf_em",
    prompt_function=arc_mcf_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    arc_challenge_cf, arc_easy_cf,
    arc_challenge_mcf, arc_easy_mcf,
    arc_challenge_mcf_em, arc_easy_mcf_em,
]
