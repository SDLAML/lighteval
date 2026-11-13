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
        query=f"Question: {line['question']}\nAnswer: ",
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


# akariasai/PopQA: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"


TASKS_TABLE = [
    LightevalTaskConfig(
        name="popqa:gen",
        prompt_function=popqa_gen_prompt,
        hf_repo="akariasai/PopQA",
        hf_revision=_REVISION,
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=128,
        # official SQuAD-style EM / F1 (article removal, Counter F1, max over aliases); version 2 (was qa_f1 / qa_em)
        metrics=[Metrics.squad_official_f1, Metrics.squad_official_em],
        stop_sequence=["\n"],
        version=2,
    ),
    LightevalTaskConfig(
        name="popqa:bpb",
        prompt_function=popqa_bpb_prompt,
        hf_repo="akariasai/PopQA",
        hf_revision=_REVISION,
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
