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


def _squad_v2_adapter(line):
    return {
        "question": line["question"],
        "context": line["context"],
        "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
    }


# Keep only answerable questions (matches opt-g): both gen and bpb always have a gold.
def _answerable(line):
    return any(ans for ans in line["answers"]["text"] if len(ans) > 0)


_qa_prompt = get_qa_prompt_function(Language.ENGLISH, _squad_v2_adapter)


def squad_v2_gen_prompt(line, task_name: str = None):
    """GenQA variant: templated QA prompt, score F1/EM against all gold answers."""
    return _qa_prompt(line, task_name)


def squad_v2_bpb_prompt(line, task_name: str = None):
    """BPB variant: identical query, but score only the first gold continuation."""
    doc = _qa_prompt(line, task_name)
    if doc is None or not doc.choices:
        return None
    gold_idx = doc.gold_index[0] if isinstance(doc.gold_index, list) else doc.gold_index
    return Doc(
        task_name=task_name,
        query=doc.query,
        choices=[doc.choices[gold_idx]],
        gold_index=0,
    )


squad_v2_gen = LightevalTaskConfig(
    name="squad_v2:gen",
    prompt_function=squad_v2_gen_prompt,
    hf_repo="rajpurkar/squad_v2",
    hf_subset="squad_v2",
    hf_filter=_answerable,
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=200,
    metrics=[Metrics.qa_f1, Metrics.qa_em],
    stop_sequence=["\n", "Question:", "question:"],
    version=1,
)

squad_v2_bpb = LightevalTaskConfig(
    name="squad_v2:bpb",
    prompt_function=squad_v2_bpb_prompt,
    hf_repo="rajpurkar/squad_v2",
    hf_subset="squad_v2",
    hf_filter=_answerable,
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=[Metrics.target_bits_per_byte],
    stop_sequence=["\n", "Question:", "question:"],
    version=1,
)

TASKS_TABLE = [
    squad_v2_gen,
    squad_v2_bpb,
]
