"""
name:
MATH rBridge

dataset:
rbridge_local (lighteval/data/titaneval/r_bridge/math_traces.jsonl)

tags:
math, reasoning, rbridge
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def math_rbridge_prompt(line, task_name: str = None):
    prompt = f"Question: {line['question']}\nLet's solve step by step."
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[line["frontier_response"]],
        gold_index=0,
        specific={
            "frontier_logprobs": line["frontier_logprobs"],
            "reasoning_byte_start": line["reasoning_byte_start"],
            "reasoning_byte_end": line["reasoning_byte_end"],
        },
    )


# One task per MATH subset (filtered by task field), plus one aggregated task
TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"math_rbridge:{subset}",
        prompt_function=math_rbridge_prompt,
        hf_repo="rbridge_local",
        hf_subset="math",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        hf_filter=lambda line, s=subset: line["task"] == f"math_{s}",
        generation_size=-1,
        metrics=[Metrics.rbridge],
        stop_sequence=[],
        version=0,
    )
    for subset in _MATH_SUBSETS
] + [
    LightevalTaskConfig(
        name="math_rbridge",
        prompt_function=math_rbridge_prompt,
        hf_repo="rbridge_local",
        hf_subset="math",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=-1,
        metrics=[Metrics.rbridge],
        stop_sequence=[],
        version=0,
    )
]
