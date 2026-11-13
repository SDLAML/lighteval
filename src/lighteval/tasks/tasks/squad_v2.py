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


# Keep only answerable questions. Used by the variants that need a gold continuation (:bpb) and by
# :gen_answerable (the old answerable-only form). The official :gen keeps ALL questions (OLMES does too).
def _answerable(line):
    return any(ans for ans in line["answers"]["text"] if len(ans) > 0)


_qa_prompt = get_qa_prompt_function(Language.ENGLISH, _squad_v2_adapter)

# OLMES squad2 (oe_eval/tasks/oe_eval_tasks/squad2.py, context_kwargs): description, no-answer string, prompt layout.
# The description is `Doc.instruction`: the prompt manager puts it once at the top and joins it with "\n\n", which
# gives OLMES's `description + "\n\n"`, so it carries no trailing newlines here.
SQUAD_V2_DESCRIPTION = (
    "Answer each question using information in the preceding background paragraph. "
    'If there is not enough information provided, answer with "Not in background".'
)
SQUAD_V2_NO_ANSWER = "Not in background"


def squad_v2_gen_prompt(line, task_name: str = None):
    """Official SQuAD v2 form with the OLMES squad2 prompt: ALL questions, abstention = "Not in background".

    Query "Title: ..\\n\\nBackground: ..\\n\\nQuestion: ..\\n\\nAnswer:"; a question without answer has the single
    answer "Not in background" (what the shots show) and `specific["unanswerable"]`, which the scorer
    (`Metrics.squad_v2_official_*`) turns into the official gold "" (a prediction "Not in background" counts as "").
    """
    answers = list(dict.fromkeys(ans for ans in line["answers"]["text"] if len(ans) > 0))
    unanswerable = len(answers) == 0
    choices = [SQUAD_V2_NO_ANSWER] if unanswerable else answers
    return Doc(
        task_name=task_name,
        query=f"Title: {line['title']}\n\nBackground: {line['context']}\n\nQuestion: {line['question']}\n\nAnswer:",
        choices=choices,
        gold_index=list(range(len(choices))),
        instruction=SQUAD_V2_DESCRIPTION,
        specific={"unanswerable": unanswerable},
    )


def squad_v2_gen_answerable_prompt(line, task_name: str = None):
    """Old GenQA variant (answerable questions only): templated QA prompt, F1/EM against all gold answers."""
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


# rajpurkar/squad_v2: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "3ffb306f725f7d2ce8394bc1873b24868140c412"


# Official SQuAD v2 (all 11,873 validation questions; few-shot pool = all of train, so the shots show the abstention).
# OLMES squad2: stop ["Title:", "\n\n"], max_gen_toks 50. Scorer = official compute_score.py (see SquadOfficialScore);
# `empty_pred` logs the share of empty generations (an empty output is the official no-answer prediction).
squad_v2_gen = LightevalTaskConfig(
    name="squad_v2:gen",
    prompt_function=squad_v2_gen_prompt,
    hf_repo="rajpurkar/squad_v2",
    hf_revision=_REVISION,
    hf_subset="squad_v2",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=50,
    metrics=[Metrics.squad_v2_official_f1, Metrics.squad_v2_official_em, Metrics.empty_pred],
    stop_sequence=["Title:", "\n\n"],
    version=2,
)

# The old form: answerable questions only (5,928 of 11,873), old QA prompt, F1/EM of `qa_f1` / `qa_em`.
squad_v2_gen_answerable = LightevalTaskConfig(
    name="squad_v2:gen_answerable",
    prompt_function=squad_v2_gen_answerable_prompt,
    hf_repo="rajpurkar/squad_v2",
    hf_revision=_REVISION,
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
    hf_revision=_REVISION,
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
    squad_v2_gen_answerable,
    squad_v2_bpb,
]
