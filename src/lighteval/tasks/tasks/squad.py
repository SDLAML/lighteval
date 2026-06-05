"""
name:
SQuAD

dataset:
allenai/squad (SQuAD v1.1)

abstract:
Stanford Question Answering Dataset (SQuAD) v1.1 is a reading comprehension
dataset of questions posed by crowdworkers on Wikipedia articles, where the
answer is a span from the corresponding passage.

languages:
english

tags:
qa, reading-comprehension

paper:
https://arxiv.org/abs/1606.05250
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def squad_gen_prompt(line, task_name: str = None):
    """GenQA variant: generate answer, score with F1 against all valid answers."""
    answers_text = line["answers"]["text"]
    if not answers_text:
        return None
    is_few_shots = line.get("__few_shots", False)
    prefix = " " if is_few_shots else ""
    return Doc(
        task_name=task_name,
        query=f"Title: {line['title']}\n\nBackground: {line['context']}\n\nQuestion: {line['question']}\n\nAnswer:",
        choices=[f"{prefix}{ans}" for ans in answers_text],
        gold_index=list(range(len(answers_text))),
    )


def squad_bpb_prompt(line, task_name: str = None):
    """BPB/CF variant: score first gold answer continuation."""
    answers_text = line["answers"]["text"]
    if not answers_text:
        return None
    gold_text = answers_text[0]
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Title: {line['title']}\n\nBackground: {line['context']}\n\nQuestion: {line['question']}\n\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="squad:bpb",
        prompt_function=squad_bpb_prompt,
        hf_repo="allenai/squad",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        stop_sequence=["Title:", "\n\n"],
        metrics=[Metrics.target_bits_per_byte],
        version=1,
    ),
    LightevalTaskConfig(
        name="squad:gen",
        prompt_function=squad_gen_prompt,
        hf_repo="allenai/squad",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=50,
        stop_sequence=["Title:", "\n\n"],
        metrics=[Metrics.f1_score, Metrics.exact_match],
        version=1,
    ),
]
