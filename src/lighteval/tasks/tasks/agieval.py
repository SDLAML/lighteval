"""
name:
Agieval

dataset:
dmayhem93/agieval-aqua-rat, dmayhem93/agieval-gaokao-biology, dmayhem93/agieval-gaokao-chemistry, dmayhem93/agieval-gaokao-chinese, dmayhem93/agieval-gaokao-english, dmayhem93/agieval-gaokao-geography, dmayhem93/agieval-gaokao-history, dmayhem93/agieval-gaokao-mathqa, dmayhem93/agieval-gaokao-physics, dmayhem93/agieval-logiqa-en, dmayhem93/agieval-logiqa-zh, dmayhem93/agieval-lsat-ar, dmayhem93/agieval-lsat-lr, dmayhem93/agieval-lsat-rc, dmayhem93/agieval-sat-en, dmayhem93/agieval-sat-en-without-passage, dmayhem93/agieval-sat-math

abstract:
AGIEval is a human-centric benchmark specifically designed to evaluate the
general abilities of foundation models in tasks pertinent to human cognition and
problem-solving. This benchmark is derived from 20 official, public, and
high-standard admission and qualification exams intended for general human
test-takers, such as general college admission tests (e.g., Chinese College
Entrance Exam (Gaokao) and American SAT), law school admission tests, math
competitions, lawyer qualification tests, and national civil service exams.

languages:
english, chinese

tags:
biology, chemistry, geography, history, knowledge, language, multiple-choice, physics, reasoning

paper:
https://arxiv.org/abs/2304.06364
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def record_to_sample(record):
    # we need to remove prepended (A), (B), (C), (D) from the choices
    choices = [
        c.replace("(A)", "").replace("(B)", "").replace("(C)", "").replace("(D)", "").strip()
        for c in record["choices"]
    ]
    return Sample(input=record["query"], target=ascii_uppercase[record["gold"][0]], choices=choices)


def agieval_prompt(line, task_name: str = None):
    if line["gold"][0] >= len(line["choices"]):
        print("Warning: gold index is out of bounds for choices, skipping sample.")
        return None
    return Doc(
        task_name=task_name,
        query=line["query"].strip(),
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["gold"],
    )

def agieval_prompt_cot_eng(line, task_name: str = None):
    query = f"Answer the following multiple choice question (with answers). Think step by step before giving the final answer.\n\nQuestion: {line['question']}"
    query += "".join([f"\n{key}. {choice.replace("(A)", "").replace("(B)", "").replace("(C)", "").replace("(D)", "").replace("(E)", "").strip()}" 
                      for key, choice in zip(ascii_uppercase, line["options"])])
    query += "\nSolution:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {c}" for c in ascii_uppercase[:len(line["options"])]],
        gold_index=ascii_uppercase.index(line["label"]),
    )

AGIEVAL_SUBSETS = [
    "aqua_rat",
    # "aqua-rat",
    # "gaokao-biology",
    # "gaokao-chemistry",
    # "gaokao-chinese",
    # "gaokao-english",
    # "gaokao-geography",
    # "gaokao-history",
    # "gaokao-mathqa",
    # "gaokao-physics",
    "logiqa-en",
    # "logiqa-zh",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
    # "sat-en-without-passage",
    "sat-math",
]

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"agieval_eng_em:{subset}",
        sample_fields=record_to_sample,
        solver=[multiple_choice(cache=True)],
        scorer=choice(),
        prompt_function=agieval_prompt_cot_eng,
        hf_repo="lighteval/agi_eval_en",
        hf_subset=subset,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["train"],
        few_shots_split="validation",
        few_shots_select=None,
        generation_size=512,
        metrics=[
            # Metrics.exact_match,
            Metrics.gpqa_instruct_metric,
        ],
        stop_sequence=["Question:"],
        version=0,
    )
    for subset in AGIEVAL_SUBSETS
]

# TASKS_TABLE = [
#     LightevalTaskConfig(
#         name=f"agieval:{subset}",
#         sample_fields=record_to_sample,
#         solver=[multiple_choice(cache=True)],
#         scorer=choice(),
#         prompt_function=agieval_prompt,
#         hf_repo=f"dmayhem93/agieval-{subset}",
#         hf_subset="default",
#         hf_avail_splits=["test"],
#         evaluation_splits=["test"],
#         few_shots_split=None,
#         few_shots_select="random_sampling",
#         generation_size=1,
#         metrics=[
#             Metrics.loglikelihood_acc,
#         ],
#         stop_sequence=None,
#         version=0,
#     )
#     for subset in AGIEVAL_SUBSETS
# ]