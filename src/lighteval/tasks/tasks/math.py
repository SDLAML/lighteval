"""
name:
Math

dataset:
DigitalLearningGmbH/MATH-lighteval

abstract:

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2305.20050
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import math_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


MATH_PROMPT_TEMPLATE = """
Solve the following math problem. Think step by step before giving the final answer.

Problem:
{prompt}

Solution:
""".strip()

# OLMo easy-suite style: minimal prompt, 4-shot from training data
MATH_BPB_PROMPT_TEMPLATE = "Problem: {prompt}\nSolution:"

_MATH_SUBSETS = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus',
]

def math_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["problem"]),
        # query=f"Question: {line['problem']}\nAnswer:",
        choices=[f" {line['solution']}"],
        gold_index=0,
    )

def math_bpb_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_BPB_PROMPT_TEMPLATE.format(prompt=line["problem"]),
        choices=[f" {line['solution']}"],
        gold_index=0,
    )

# BPB variant: OLMo-style minimal prompt, 4-shot from training data
TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"math:{subset}:bpb",
        prompt_function=math_bpb_prompt,
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
        hf_subset=subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=[Metrics.target_bits_per_byte],
        stop_sequence=["Problem:"],
        version=1,
    )
    for subset in _MATH_SUBSETS
]
