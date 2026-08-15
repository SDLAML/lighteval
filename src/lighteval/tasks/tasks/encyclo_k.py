"""
name:
Encyclo-K

dataset:
m-a-p/Encyclo-K

abstract:
Encyclo-K is a statement-based multiple-choice benchmark built from dynamically
composed knowledge statements across 11 disciplines.

languages:
english, chinese

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2512.24867
"""

import re
from functools import partial
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


_ENCYCLO_K_DISCIPLINES = [
    "Agriculture",
    "Economics",
    "Education",
    "Engineering",
    "History",
    "Law",
    "Literature",
    "Management",
    "Medicine",
    "Philosophy",
    "Science",
]

_HAN_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
    r"\U00020000-\U0002EBEF\U00030000-\U0003134F]"
)


def encyclo_k_prompt(line, task_name: str = None):
    """Use the prompt supplied by Encyclo-K and score its final answer letter."""
    choices = list(ascii_uppercase[: len(line["options"])])

    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=choices,
        gold_index=choices.index(line["answer_letter"]),
        instruction=line["prompt"],
    )


def _has_discipline(line, discipline: str):
    return line["discipline"] == discipline


def _has_english_discipline(line, discipline: str):
    return _has_discipline(line, discipline) and _HAN_RE.search(line["file_name"]) is None


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"encyclo_k:{discipline.lower()}",
        prompt_function=encyclo_k_prompt,
        hf_repo="m-a-p/Encyclo-K",
        hf_subset="default",
        hf_revision="0f81da9652b9c4d7bad73ce9576d81901d9441f2",
        hf_filter=partial(_has_discipline, discipline=discipline),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=[],
        version=0,
    )
    for discipline in _ENCYCLO_K_DISCIPLINES
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"encyclo_k_en:{discipline.lower()}",
        prompt_function=encyclo_k_prompt,
        hf_repo="m-a-p/Encyclo-K",
        hf_subset="default",
        hf_revision="0f81da9652b9c4d7bad73ce9576d81901d9441f2",
        hf_filter=partial(_has_english_discipline, discipline=discipline),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=[],
        version=0,
    )
    for discipline in _ENCYCLO_K_DISCIPLINES
]
