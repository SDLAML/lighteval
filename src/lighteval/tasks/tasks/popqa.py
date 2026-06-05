"""
name:
PopQA

dataset:
akariasai/PopQA

abstract:
PopQA is an open-domain question answering dataset of entity-centric questions,
designed to probe the factual knowledge of language models across a range of
entity popularities. Each question has a set of acceptable answer aliases.

languages:
english

tags:
qa, factuality

paper:
https://arxiv.org/abs/2212.10511
"""

import ast

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def _answers(line):
    return ast.literal_eval(line["possible_answers"])


def popqa_gen_prompt(line, task_name: str = None):
    """GenQA variant: generate answer, score F1/EM against all acceptable answers."""
    answers = _answers(line)
    # query already ends with a space, so the gold continuation needs no extra prefix
    return Doc(
        task_name=task_name,
        query=f"{line['question']} ",
        choices=list(answers),
        gold_index=list(range(len(answers))),
    )


def popqa_bpb_prompt(line, task_name: str = None):
    """BPB variant: score the first acceptable answer as the gold continuation."""
    answers = _answers(line)
    if not answers:
        return None
    return Doc(
        task_name=task_name,
        query=f"{line['question']} ",
        choices=[answers[0]],
        gold_index=0,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="popqa:gen",
        prompt_function=popqa_gen_prompt,
        hf_repo="akariasai/PopQA",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=8,
        metrics=[Metrics.f1_score, Metrics.exact_match],
        stop_sequence=["\n"],
        version=1,
    ),
    LightevalTaskConfig(
        name="popqa:bpb",
        prompt_function=popqa_bpb_prompt,
        hf_repo="akariasai/PopQA",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=-1,
        metrics=[Metrics.target_bits_per_byte],
        stop_sequence=["\n"],
        version=1,
    ),
]
