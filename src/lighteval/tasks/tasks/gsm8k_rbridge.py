"""
name:
GSM8K rBridge

dataset:
rbridge_local (lighteval/data/titaneval/rbridge_data/gsm8k_traces.jsonl)

tags:
math, reasoning, rbridge
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def gsm8k_rbridge_prompt(line, task_name: str = None):
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


TASKS_TABLE = [
    LightevalTaskConfig(
        name="gsm8k_rbridge",
        prompt_function=gsm8k_rbridge_prompt,
        hf_repo="rbridge_local",
        hf_subset="gsm8k",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=-1,
        metrics=[Metrics.rbridge],
        stop_sequence=[],
        version=0,
    )
]
