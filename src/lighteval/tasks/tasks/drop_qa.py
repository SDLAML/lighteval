"""
name:
Drop Qa

dataset:
lighteval/drop_harness

abstract:
The DROP dataset is a new question-answering dataset designed to evaluate the
ability of language models to answer complex questions that require reasoning
over multiple sentences.

languages:
english

tags:
math, qa, reasoning

paper:
https://arxiv.org/abs/1810.00505
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def drop_gen_prompt(line, task_name: str = None):
    """GenQA variant: CF-style prompt, generate answer, score with F1."""
    answer = line["answer"]
    if answer["number"] != "":
        gold_text = str(answer["number"])
    elif answer["spans"]:
        gold_text = answer["spans"][0]
    else:
        gold_text = " ".join([answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]).strip()
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {line['question']}\nAnswer:",
        choices=[f"{' ' if is_few_shots else ''}{gold_text}"],
        gold_index=0,
    )


def drop_bpb_prompt(line, task_name: str = None):
    """BPB variant: CF-style prompt with only the first gold answer."""
    answer = line["answer"]
    if answer["number"] != "":
        gold_text = str(answer["number"])
    elif answer["spans"]:
        gold_text = ", ".join(answer["spans"])
    else:
        gold_text = " ".join([answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]).strip()
    if not gold_text:
        return None  # degenerate sample with no extractable answer; skip
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {line['question']}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )

drop_bpb = LightevalTaskConfig(
    name="drop:bpb",
    prompt_function=drop_bpb_prompt,
    hf_repo="lighteval/drop_harness",
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    stop_sequence=["\n"],
    metrics=[Metrics.target_bits_per_byte],
    version=0,
)

drop_gen = LightevalTaskConfig(
    name="drop:gen",
    prompt_function=drop_gen_prompt,
    hf_repo="lighteval/drop_harness",
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=100,
    stop_sequence=["\n"],
    metrics=[Metrics.drop],
    version=0,
)

TASKS_TABLE = [
    drop_bpb,
    drop_gen,
]
