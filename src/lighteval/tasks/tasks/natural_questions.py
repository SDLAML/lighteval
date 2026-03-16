"""
name:
Natural Questions

dataset:
lighteval/small_natural_questions

abstract:
This dataset is a collection of question-answer pairs from the Natural Questions
dataset. See Natural Questions for additional information. This dataset can be
used directly with Sentence Transformers to train embedding models.

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
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


def nq_bpb_prompt(line, task_name: str = None):
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


natural_questions_bpb = LightevalTaskConfig(
    name="natural_questions:bpb",
    prompt_function=nq_bpb_prompt,
    hf_repo="lighteval/small_natural_questions",
    hf_subset="default",
    evaluation_splits=("test",),
    few_shots_split="few_shot",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    stop_sequence=["\n"],
    metrics=[Metrics.target_bits_per_byte],
    version=0,
)

TASKS_TABLE = [
    natural_questions_bpb,
]
