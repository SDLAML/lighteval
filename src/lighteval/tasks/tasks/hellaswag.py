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


def harness_preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def hellaswag_prompt(line, task_name: str = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n\n"
    query += f"Question: {line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["endings"])])
    query += "Answer:"

    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in ascii_uppercase[: len(line["endings"])]],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n\n",
    )

def hellaswag_harness_prompt(line, task_name: str = None):
    ctx = line["ctx_a"] + " " + line["ctx_b"].capitalize()
    query = harness_preprocess(line["activity_label"] + ": " + ctx)
    choices = [harness_preprocess(ending) for ending in line["endings"]]
    choices = [c if c[:1].isspace() else " " + c for c in choices]
    gold_ix = int(line["label"]) if str(line.get("label", "")).strip() != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_ix,
    )

hellaswag = LightevalTaskConfig(
    name="hellaswag",
    prompt_function=hellaswag_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

hellaswag_harness = LightevalTaskConfig(
    name="hellaswag_harness",
    prompt_function=hellaswag_harness_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    # Optional but recommended for reproducibility - specify the exact commit hash:
    # hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
    # trust_dataset=True,
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation",],
    metrics=[
        LogLikelihoodAccMetric(),
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        ],
)

TASKS_TABLE = [
    hellaswag,
    hellaswag_harness,
]
