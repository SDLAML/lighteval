"""
name:
Simpleqa

dataset:
lighteval/SimpleQA

abstract:
A factuality benchmark called SimpleQA that measures the ability for language
models to answer short, fact-seeking questions.

languages:
english

tags:
factuality, general-knowledge, qa

paper:
https://openai.com/index/introducing-simpleqa/

starred:
true
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def record_to_sample(record):
    query = record["problem"]
    target = record["answer"]
    return Sample(input=query, target=target)


# ---- Original graded variant (kept for compatibility) ----

def simpleqa_prompt(line, task_name: str = None):
    query = f"Question: {line['question']}\n"
    query += "".join(
        [f"\n{key}. {choice}" for key, choice in zip(["A", "B", "C", "D", "E", "F"], line["choices"]["text"])]
    )
    query += "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=line["choices"]["text"],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


simpleqa = LightevalTaskConfig(
    name="simpleqa",
    prompt_function=simpleqa_prompt,
    hf_repo="lighteval/SimpleQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="few_shot",
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=model_graded_fact(),
)

# ---- GenQA variants (our convention: gen{em,f1} + decoupled bpb) ----

def simpleqa_gen_prompt(line, task_name: str = None):
    """GenQA variant: generate short answer, score with F1/EM."""
    answer = line["answer"]
    prefix = " " if line.get("__few_shots", False) else ""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['problem']}\nAnswer:",
        choices=[f"{prefix}{answer}"],
        gold_index=0,
    )


def simpleqa_bpb_prompt(line, task_name: str = None):
    """BPB variant: score the gold answer continuation."""
    answer = line["answer"]
    if not answer:
        return None
    if not answer[0].isspace():
        answer = " " + answer
    return Doc(
        task_name=task_name,
        query=f"Question: {line['problem']}\nAnswer:",
        choices=[answer],
        gold_index=0,
    )


simpleqa_gen = LightevalTaskConfig(
    name="simpleqa:gen",
    prompt_function=simpleqa_gen_prompt,
    hf_repo="lighteval/SimpleQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="few_shot",
    few_shots_select="random_sampling",
    generation_size=50,
    metrics=[Metrics.qa_f1, Metrics.qa_em],
    stop_sequence=["\n"],
    version=1,
)

simpleqa_bpb = LightevalTaskConfig(
    name="simpleqa:bpb",
    prompt_function=simpleqa_bpb_prompt,
    hf_repo="lighteval/SimpleQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="few_shot",
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=[Metrics.target_bits_per_byte],
    stop_sequence=["\n"],
    version=1,
)

TASKS_TABLE = [
    simpleqa,
    simpleqa_gen,
    simpleqa_bpb,
]
