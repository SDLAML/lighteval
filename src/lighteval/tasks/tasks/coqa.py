"""
name:
Coqa

dataset:
stanfordnlp/coqa

abstract:
CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.

languages:
english

tags:
dialog, qa

paper:
https://arxiv.org/abs/1808.07042
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def coqa_gen_prompt(line, task_name: str = None):
    """GenQA variant: first Q only, generate answer, score with F1. 0-shot."""
    q = line["questions"][0]
    a = line["answers"]["input_text"][0]
    if not a:
        return None
    return Doc(
        task_name=task_name,
        query=f"{line['story']}\n\nQ: {q}\nA:",
        choices=[f" {a}"],
        gold_index=0,
    )


def coqa_bpb_prompt(line, task_name: str = None):
    """BPB variant: first question only, gold answer as single choice."""
    q = line["questions"][0]
    a = line["answers"]["input_text"][0]
    if not a:
        return None
    gold_text = a if a[0].isspace() else " " + a
    return Doc(
        task_name=task_name,
        query=f"{line['story']}\n\nQ: {q}\nA:",
        choices=[gold_text],
        gold_index=0,
    )


coqa_bpb = LightevalTaskConfig(
    name="coqa:bpb",
    prompt_function=coqa_bpb_prompt,
    hf_repo="stanfordnlp/coqa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.target_bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

coqa_gen = LightevalTaskConfig(
    name="coqa:gen",
    prompt_function=coqa_gen_prompt,
    hf_repo="stanfordnlp/coqa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=50,
    stop_sequence=["\n"],
    metrics=[Metrics.f1_score, Metrics.exact_match],
    version=0,
)

TASKS_TABLE = [
    coqa_bpb,
    coqa_gen,
]
