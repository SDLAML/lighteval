"""
name:
SimpleQA Verified (English), base-model protocol (short-answer generation)

dataset:
google/simpleqa-verified

abstract:
The English original of the 1,000 SimpleQA Verified questions (Haas et al. 2025, arXiv 2509.07968; google/simpleqa-verified, config
simpleqa_verified, split eval, MIT) in the format used for base-model checkpoints, the ENGLISH TWIN of simpleqa_verified_base:<lang>:gen
(simpleqa_verified_base.py, eight European languages, no English): the same "Question: ... / Answer:" cloze prompt (the English literals of the
fork's template, as the twin builds it for its languages), the same five demonstrations, at most 128 new tokens, stop at newline, and the fork's
quasi exact match (em) / quasi F1 (f1) of the twin (same normaliser, no judge).
The five demonstrations are the five questions that the twin holds out (simpleqa_verified_base.SHOT_IDS, ids simpleqa_verified_<original_index>):
the English source rows of those ids are found by `original_index` (the translated dataset's id is "simpleqa_verified_" + the source row's
original_index) and are held out of the English eval split too: 995 of 1,000 questions are scored.
Differences to the twin that come from the data: the English split has no answer aliases (the translated one has aliases written by the translator),
and its column names are problem / answer. The acceptance-range note that the original appends to the golds of the numeric questions
("120,000 (acceptable range: anything between 118,000 and 122,000)": 88 of the 1,000 rows, one of them demonstration 1202) is removed from the
gold before it is scored or shown, as the twin does for its translated forms of the note (version 1 there).
The task name has the twin's three-part form, but it is NOT part of the superset `simpleqa_verified_base:gen` (`superset_keys=()`): that spec keeps
running the eight non-English languages only.

languages:
english

tags:
factuality, general-knowledge, qa

paper:
https://arxiv.org/abs/2509.07968
"""

import re

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.simpleqa_verified import _get_qa_prompt_function
from lighteval.tasks.tasks.heldout_shots import rows_by_key
from lighteval.tasks.tasks.simpleqa_verified_base import SHOT_IDS, _metrics
from lighteval.utils.language import Language


REPO = "google/simpleqa-verified"
SUBSET = "simpleqa_verified"
REVISION = "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc"  # the pin of the chat task simpleqa_verified_chat (chat_judged.py); card license MIT; 1,000 rows, split eval
TASK_NAME = "simpleqa_verified_base:en:gen"


def row_id(row: dict) -> str:
    """The twin's id of an English source row: ellamind/simpleqa-verified-multilingual ids are "simpleqa_verified_<original_index>" (checked for all rows of all 8 languages)."""
    return f"simpleqa_verified_{int(row['original_index'])}"


# The numeric golds carry the original grader's acceptance range, e.g. "390 metres (acceptable range: anything between 386 and 394 metres)" (88 rows of the pinned
# split). An exact-match scorer can never match such a gold, and demonstration 1202 would teach the model to add a note. The note is removed from the gold (the twin
# does the same for the translated forms, simpleqa_verified_base.strip_range_note); other parentheses are part of real answers ("IIT (BHU)") and stay.
_RANGE_NOTE = re.compile(r"\s*\(acceptable range\b[^)]*\)\s*$", re.IGNORECASE)


def strip_range_note(text: str) -> str:
    return _RANGE_NOTE.sub("", text) if isinstance(text, str) else text


def _as_twin_row(line: dict) -> dict:
    """The row in the form the twin's prompt function reads: question / answer / answer_aliases (this split has no aliases)."""
    return {"question": line["problem"], "answer": strip_range_note(line["answer"]), "answer_aliases": []}


def _prompt_function():
    base = _get_qa_prompt_function(Language.ENGLISH, include_aliases=True)

    def prompt(line, task_name: str = None):
        if not line.get("__few_shots") and row_id(line) in SHOT_IDS:
            return None  # held out: a demonstration only
        return base(_as_twin_row(line), task_name)

    return prompt


def shot_rows() -> list[dict]:
    """The held-out English rows, in the order the twin shows them."""
    return rows_by_key(REPO, SUBSET, "eval", REVISION, row_id, SHOT_IDS)


TASKS_TABLE = [
    LightevalTaskConfig(
        name=TASK_NAME,
        prompt_function=_prompt_function(),
        hf_repo=REPO,
        hf_subset=SUBSET,
        hf_revision=REVISION,
        hf_avail_splits=["eval"],
        evaluation_splits=["eval"],
        few_shots_rows=shot_rows,
        few_shots_select="sequential",
        generation_size=128,
        stop_sequence=["\n"],
        metrics=_metrics(Language.ENGLISH),   # em / f1, the twin's scorers and normaliser
        superset_keys=(),                     # in no superset: `simpleqa_verified_base:gen` stays the eight non-English languages
        version=1,                            # the twin's protocol version 1 (range notes removed); there is no earlier English version
    )
]
