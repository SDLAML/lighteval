"""
name:
Gsm8K

dataset:
openai/gsm8k

abstract:
GSM8K is a dataset of 8,000+ high-quality, single-step arithmetic word problems.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2110.14168
"""

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# setup for problem + instructions for providing answer
MATH_PROMPT_TEMPLATE = """
Solve the following math problem. Think step by step before giving the final answer.

Problem:
{prompt}

Solution:
""".strip()


def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample):
    return f"{sample.input}\n\nReasoning:\n" + f"{sample.metadata['reasoning']}\n\n" + f"ANSWER: {sample.target}"


def final_number(answer: str) -> str:
    """The gold of an item: the text after the last '####' with the thousands commas removed (lm-evaluation-harness:
    `answer.split('#### ')[-1]`). The metric must see ONLY this: given the whole worked solution, the extractor's 'answer'
    anchor (extractive_match_utils.py, priority 200) fires on reasoning text and picked a wrong gold for 4 of 1,319 items."""
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_prompt(line, task_name: str = None):
    if line.get("__few_shots"):
        answer = line["answer"].split("####")
        target = answer.pop().strip()
        reasoning = "####".join(answer).strip()
        exemplar = reasoning if reasoning else ""
        if exemplar:
            exemplar += f"\n\nANSWER: {target}"
        else:
            exemplar = target
        return Doc(
            task_name=task_name,
            query=MATH_PROMPT_TEMPLATE.format(prompt=line["question"]),
            choices=[exemplar],
            gold_index=0,
        )

    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["question"]),
        choices=[final_number(line["answer"])],
        gold_index=0,
    )


# openai/gsm8k: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "740312add88f781978c0658806c59bc2815b9866"


gsm8k = LightevalTaskConfig(
    name="gsm8k",
    prompt_function=gsm8k_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="openai/gsm8k",
    hf_revision=_REVISION,
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=512,
    metrics=[
        Metrics.expr_gold_metric,
    ],
    stop_sequence=["Question:", "\n####", "####"],
    version=1,                      # 1: gold = the final number only (was the whole solution)
)

TASKS_TABLE = [
    gsm8k,
]
