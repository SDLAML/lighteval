"""
name:
Squad V2

dataset:
rajpurkar/squad_v2

abstract:
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset,
consisting of questions posed by crowdworkers on a set of Wikipedia articles,
where the answer to every question is a segment of text, or span, from the
corresponding reading passage, or the question might be unanswerable.
SQuAD 2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000
unanswerable questions written adversarially by crowdworkers to look similar to
answerable ones. To do well on SQuAD2.0, systems must not only answer questions
when possible, but also determine when no answer is supported by the paragraph
and abstain from answering.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1806.03822
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


def squad_bpb_prompt(line, task_name: str = None):
    """BPB variant: CF-style prompt with first valid gold answer as single choice."""
    valid_answers = [ans for ans in line["answers"]["text"] if len(ans) > 0]
    if not valid_answers:
        return None  # unanswerable question; skip
    gold_text = valid_answers[0]
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Context: {line['context']}\nQuestion: {line['question']}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )

squad_v2_em = LightevalTaskConfig(
    name="squad_v2:em",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "context": line["context"],
            "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
        },
    ),
    hf_repo="rajpurkar/squad_v2",
    hf_subset="squad_v2",
    hf_filter=lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    evaluation_splits=("validation",),
    few_shots_split="train",
    stop_sequence=["\n", "Question:", "question:"],
    generation_size=200,
    metrics=[Metrics.exact_match],
    version=1,
)

squad_v2_bpb = LightevalTaskConfig(
    name="squad_v2:bpb",
    prompt_function=squad_bpb_prompt,
    hf_repo="rajpurkar/squad_v2",
    hf_subset="squad_v2",
    hf_filter=lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    evaluation_splits=("validation",),
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    stop_sequence=["\n"],
    metrics=[Metrics.target_bits_per_byte],
    version=0,
)

TASKS_TABLE = [
    squad_v2_em,
    squad_v2_bpb,
]
