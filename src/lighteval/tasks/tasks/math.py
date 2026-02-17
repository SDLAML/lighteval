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

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"math:{subset}",
        prompt_function=math_prompt,
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
        hf_subset=subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1024,
        metrics=[
            # Metrics.maj_at_n(
            #     sample_params={
            #         "n": 4,
            #         "strip_strings": True,
            #         "normalize_pred": math_normalizer,
            #         "normalize_gold": math_normalizer,
            #     }
            # ),
            Metrics.expr_gold_metric,
        ],
        stop_sequence=["Question:", "Problem:"],
        version=1,
    )
    for subset in _MATH_SUBSETS
]
