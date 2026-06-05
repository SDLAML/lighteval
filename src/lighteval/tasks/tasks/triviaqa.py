"""
name:
Triviaqa

dataset:
mandarjoshi/trivia_qa

abstract:
TriviaqQA is a reading comprehension dataset containing over 650K
question-answer-evidence triples. TriviaqQA includes 95K question-answer pairs
authored by trivia enthusiasts and independently gathered evidence documents,
six per question on average, that provide high quality distant supervision for
answering the questions.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1705.03551
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_QUERY = "Question: {question}\nAnswer:"


def _gold_answers(line):
    """Canonical value first (nice few-shot rendering), then aliases for F1/EM coverage."""
    golds = [line["answer"]["value"]]
    for alias in line["answer"]["aliases"]:
        if alias not in golds:
            golds.append(alias)
    return golds


def triviaqa_gen_prompt(line, task_name: str = None):
    """GenQA variant: generate answer, score F1/EM against all valid answers."""
    golds = _gold_answers(line)
    prefix = " " if line.get("__few_shots", False) else ""
    return Doc(
        task_name=task_name,
        query=_QUERY.format(question=line["question"]),
        choices=[f"{prefix}{g}" for g in golds],
        gold_index=list(range(len(golds))),
    )


def triviaqa_bpb_prompt(line, task_name: str = None):
    """BPB variant: score the canonical gold continuation."""
    gold_text = line["answer"]["value"]
    if not gold_text:
        return None
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=_QUERY.format(question=line["question"]),
        choices=[gold_text],
        gold_index=0,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="triviaqa:gen",
        prompt_function=triviaqa_gen_prompt,
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=20,
        metrics=[Metrics.f1_score, Metrics.exact_match],
        stop_sequence=["\n", ".", ","],
        version=1,
    ),
    LightevalTaskConfig(
        name="triviaqa:bpb",
        prompt_function=triviaqa_bpb_prompt,
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=[Metrics.target_bits_per_byte],
        stop_sequence=["\n", ".", ","],
        version=1,
    ),
]
