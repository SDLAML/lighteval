"""
name:
CoQA

dataset:
EleutherAI/coqa (refs/convert/parquet)

abstract:
CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.

Note: Each HF row is a full story with multiple turns. We evaluate all turns by
building the full preceding-QA context for each turn, using the first answer as
the reference for all prior turns. This matches the OLMES multi-turn setup.

languages:
english

tags:
dialog, qa

paper:
https://arxiv.org/abs/1808.07042
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def _build_query(story: str, questions: list, answers: list, turn_idx: int) -> str:
    """Build OLMES-style multi-turn query for a given turn."""
    query = f"Passage: {story}"
    if turn_idx > 0:
        query += "\n\nPreceding questions:"
        for i in range(turn_idx):
            query += f"\n\nQuestion: {questions[i]}\nAnswer: {answers[i]}"
    query += "\n\nFinal question:"
    query += f"\n\nQuestion: {questions[turn_idx]}\nAnswer:"
    return query


def coqa_gen_prompt(line, task_name: str = None):
    """GenQA: evaluate ALL turns, build full preceding-QA context per turn.

    Returns a list of Docs (one per turn), matching OLMO's CoQA._process_doc_to_multi.
    The lighteval_task.py extension (list-of-Docs support) handles multi-doc rows.
    """
    questions = line["questions"]["input_text"]
    answers = line["answers"]["input_text"]
    if not questions or not answers:
        return None
    is_few_shots = line.get("__few_shots", False)
    docs = []
    for turn_idx in range(len(questions)):
        gold_text = answers[turn_idx]
        if not gold_text:
            continue
        docs.append(Doc(
            task_name=task_name,
            query=_build_query(line["story"], questions, answers, turn_idx),
            choices=[f"{' ' if is_few_shots else ''}{gold_text}"],
            gold_index=0,
        ))
    return docs if docs else None


def coqa_bpb_prompt(line, task_name: str = None):
    """BPB: evaluate ALL turns of a story, returning one Doc per turn.

    Matches OLMO which creates one evaluation instance per (story, turn) pair.
    The full preceding-QA context is built for each turn.
    """
    questions = line["questions"]["input_text"]
    answers = line["answers"]["input_text"]
    if not questions or not answers:
        return None

    docs = []
    for turn_idx in range(len(questions)):
        gold_text = answers[turn_idx]
        if not gold_text:
            continue
        if not gold_text[0].isspace():
            gold_text = " " + gold_text
        docs.append(Doc(
            task_name=task_name,
            query=_build_query(line["story"], questions, answers, turn_idx),
            choices=[gold_text],
            gold_index=0,
        ))
    return docs if docs else None


TASKS_TABLE = [
    LightevalTaskConfig(
        name="coqa:bpb",
        prompt_function=coqa_bpb_prompt,
        hf_repo="EleutherAI/coqa",
        hf_subset="default",
        hf_revision="refs/convert/parquet",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=["\n\n"],
        metrics=[Metrics.target_bits_per_byte],
        version=1,
    ),
    LightevalTaskConfig(
        name="coqa:gen",
        prompt_function=coqa_gen_prompt,
        hf_repo="EleutherAI/coqa",
        hf_subset="default",
        hf_revision="refs/convert/parquet",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=50,
        stop_sequence=["\n\n"],
        metrics=[Metrics.f1_score, Metrics.exact_match],
        version=1,
    ),
]
