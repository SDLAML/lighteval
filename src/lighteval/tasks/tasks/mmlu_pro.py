"""
name:
MMLU Pro

dataset:
TIGER-Lab/MMLU-Pro

abstract:
MMLU-Pro dataset is a more robust and challenging massive multi-task
understanding dataset tailored to more rigorously benchmark large language
models' capabilities. This dataset contains 12K complex questions across various
disciplines.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2406.01574

starred:
true
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

{choices}

Answer:""".strip()


def mmlu_pro_mcf_prompt_function(line, task_name: str = None):
    query = f"Answer the following multiple choice question.\n\nQuestion: {line['question'].strip()}"
    query += "".join([f"\n {key}. {choice}" for key, choice in zip(ascii_uppercase, line["options"])])
    query += "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + c for c in ascii_uppercase[:len(line["options"])]],
        gold_index=line["answer_index"],
    )

def mmlu_pro_cf_prompt_function(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query="Question: " + line["question"].strip() + "\nAnswer:",
        choices= [c if c and c[0].isspace() else " " + c for c in line["options"]],
        gold_index=line["answer_index"],
    )

def mmlu_pro_prompt_function(line, task_name: str = None):
    choices = "\n".join([f"{letter}: {choice}" for letter, choice in zip(ascii_uppercase, line["options"])])

    query = TEMPLATE.format(
        question=line["question"],
        choices=choices,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase[: len(line["options"])]),
        gold_index=line["answer_index"],
        instruction=query,
    )


def record_to_sample(record):
    return Sample(input=record["question"], target=record["answer"], choices=record["options"])


_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]

mmlu_pro_cot = LightevalTaskConfig(
    name="mmlu_pro:cot",
    prompt_function=mmlu_pro_prompt_function,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split="validation",
    generation_size=32768,
    metrics=[Metrics.gpqa_instruct_metric],
)

mmlu_pro_mcf_em = LightevalTaskConfig(
    name="mmlu_pro:mcf_em",
    prompt_function=mmlu_pro_mcf_prompt_function,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split="validation",
    generation_size=1,
    stop_sequence=["\n"],
    metrics=[Metrics.exact_match],
)

mmlu_pro_mcf = LightevalTaskConfig(
    name="mmlu_pro:mcf",
    prompt_function=mmlu_pro_mcf_prompt_function,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split="validation",
    generation_size=-1,
    metrics=_MCF_METRICS,
)

mmlu_pro_cf = LightevalTaskConfig(
    name="mmlu_pro:cf",
    prompt_function=mmlu_pro_cf_prompt_function,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split="validation",
    generation_size=-1,
    metrics=_CF_METRICS,
)

TASKS_TABLE = [mmlu_pro_cot, mmlu_pro_cf, mmlu_pro_mcf, mmlu_pro_mcf_em]
