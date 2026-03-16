"""
name:
Jeopardy

dataset:
openaccess-ai-collective/jeopardy

abstract:
Jeopardy is a dataset of questions and answers from the Jeopardy game show.

languages:
english

tags:
knowledge, qa

paper:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


def jeopardy_bpb_prompt(line, task_name: str = None):
    """BPB variant: CF-style prompt with gold answer as single choice."""
    gold_text = line["answer"]
    if not gold_text:
        return None
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )


jeopardy_bpb = LightevalTaskConfig(
    name="jeopardy:bpb",
    prompt_function=jeopardy_bpb_prompt,
    hf_repo="openaccess-ai-collective/jeopardy",
    hf_subset="default",
    evaluation_splits=("train",),
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    stop_sequence=["\n"],
    metrics=[Metrics.target_bits_per_byte],
    version=0,
)

TASKS_TABLE = [
    jeopardy_bpb,
]
