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
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


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


def record_to_sample(record):
    query = record["question"].strip()
    target = record["answerKey"]
    choices = record["choices"]["text"]

    return Sample(input=query, target=target, choices=choices)


arc_challenge = LightevalTaskConfig(
    name="arc:challenge",
    prompt_function=arc_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    # generation_size=1,
    metrics=[
        # Metrics.loglikelihood_acc,
        LogLikelihoodAccMetric(),
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    ],
    stop_sequence=["\n"],
    version=0,
    # sample_fields=record_to_sample,
    # solver=[multiple_choice(cache=True)],
    # scorer=choice(),
)

arc_easy = LightevalTaskConfig(
    name="arc:easy",
    prompt_function=arc_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    # generation_size=1,
    metrics=[
        # Metrics.loglikelihood_acc,
        LogLikelihoodAccMetric(),
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    ],
    stop_sequence=["\n"],
    version=0,
    # sample_fields=record_to_sample,
    # solver=[multiple_choice(cache=True)],
    # scorer=choice(),
)

arc_challenge_mcf = LightevalTaskConfig(
    name="arc:challenge:mcf",
    prompt_function=arc_mcf_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
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
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    arc_challenge, arc_easy,
    arc_challenge_mcf, arc_easy_mcf,
]
