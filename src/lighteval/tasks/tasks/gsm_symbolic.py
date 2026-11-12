"""
name:
GSM Symbolic

dataset:
apple/GSM-Symbolic

abstract:
GSM-Symbolic is a benchmark for evaluating mathematical reasoning in LLMs,
created by Apple. It contains mathematically equivalent variants of GSM8K
problems across three difficulty levels: main (same difficulty as GSM8K),
p1 (one extra reasoning step), and p2 (two extra reasoning steps).

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2410.05229
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


_MATH_PROMPT_TEMPLATE = """Solve the following math problem. Think step by step before giving the final answer.

Problem:
{prompt}

Solution:""".strip()


def final_number(answer: str) -> str:
    """The gold of an item: the text after the last '####', thousands commas removed (as gsm8k.py). The metric must see only
    this: with the whole worked solution as gold, the extractor's 'answer' anchor fires on reasoning text (50 of 2,500 p2 items)."""
    return answer.split("####")[-1].strip().replace(",", "")


def gsm_symbolic_prompt(line, task_name: str = None):
    """8-shot CoT: same format as GSM8K — split on '####' to get reasoning + final answer."""
    if line.get("__few_shots"):
        parts = line["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        exemplar = reasoning + f"\n\nANSWER: {target}" if reasoning else target
        return Doc(
            task_name=task_name,
            query=_MATH_PROMPT_TEMPLATE.format(prompt=line["question"]),
            choices=[exemplar],
            gold_index=0,
        )
    return Doc(
        task_name=task_name,
        query=_MATH_PROMPT_TEMPLATE.format(prompt=line["question"]),
        choices=[final_number(line["answer"])],
        gold_index=0,
    )


_SUBSETS = ["main", "p1", "p2"]

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"gsm_symbolic:{subset}",
        prompt_function=gsm_symbolic_prompt,
        hf_repo="apple/GSM-Symbolic",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=512,
        metrics=[Metrics.expr_gold_metric],
        stop_sequence=["Problem:", "\n####", "####"],
        version=1,                  # 1: gold = the final number only (was the whole solution)
    )
    for subset in _SUBSETS
]
