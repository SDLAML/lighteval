"""
name:
Natural Questions

dataset:
google-research-datasets/nq_open

abstract:
Natural Questions Open is the open-domain version of Natural Questions.
Each example has a question and a list of valid answer strings.

languages:
english

tags:
general-knowledge, qa

paper:
https://ai.google.com/research/NaturalQuestions
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def nq_gen_prompt(line, task_name: str = None):
    """GenQA variant: generate answer, score with F1 against all valid answers."""
    answers = line["answer"]  # list of valid answer strings
    if not answers:
        return None
    is_few_shots = line.get("__few_shots", False)
    prefix = " " if is_few_shots else ""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f"{prefix}{ans}" for ans in answers],
        gold_index=list(range(len(answers))),
    )


def nq_bpb_prompt(line, task_name: str = None):
    """BPB/CF variant: score first valid gold answer."""
    answers = line["answer"]  # list of valid answer strings
    if not answers:
        return None
    gold_text = answers[0]
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )


# google-research-datasets/nq_open: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "5dd9790a83002ad084ddeb7c420dc716852c6f28"


TASKS_TABLE = [
    LightevalTaskConfig(
        name="natural_questions:bpb",
        prompt_function=nq_bpb_prompt,
        hf_repo="google-research-datasets/nq_open",
        hf_revision=_REVISION,
        hf_subset="nq_open",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        stop_sequence=["Question:", "Q:", "\n\n"],
        metrics=[Metrics.target_bits_per_byte],
        version=1,
    ),
    LightevalTaskConfig(
        name="natural_questions:gen",
        prompt_function=nq_gen_prompt,
        hf_repo="google-research-datasets/nq_open",
        hf_revision=_REVISION,
        hf_subset="nq_open",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=50,
        stop_sequence=["Question:", "Q:", "\n\n"],
        # official SQuAD-style EM / F1 (article removal, Counter F1, max over the gold answers); version 2 (was qa_f1 / qa_em)
        metrics=[Metrics.squad_official_f1, Metrics.squad_official_em],
        version=2,
    ),
]
