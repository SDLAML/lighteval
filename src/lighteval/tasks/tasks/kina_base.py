"""
name:
KINA, base-model protocol (mcf_em)

dataset:
2077AIDataFoundation/KINA

abstract:
KINA (899 graduate-level, ten-option questions, 12 disciplines) in the format used for base-model checkpoints, namely
five fixed demonstrations, the prompt lists the options with letters and ends "Answer:", the model writes ONE letter
(max 5 new tokens, stop at newline) and the score is the exact match of that letter (no letter is searched in the text).
KINA has only a test split, so five test questions (one per distinct gold letter, chosen by a fixed sha256 rule, see
heldout_shots.py) are held out as the demonstrations and are not scored: 894 questions are scored. One pooled task over all
12 disciplines (the discipline is kept in doc.specific for a breakdown from the details). Oleg Filatov's kina:* tasks in
kina.py are unchanged.

languages:
english

tags:
graduate-level, knowledge, multiple-choice, qa

paper:
https://arxiv.org/abs/2606.05104
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.heldout_shots import rows_by_key
from lighteval.tasks.tasks.kina import _LABELS, _option_texts


REPO = "2077AIDataFoundation/KINA"
REVISION = "860c64e856375ed42929737fb8b8f37272c97065"  # main on 2026-11-13 (card: license odc-by)
HEADER = "The following are multiple choice questions (with answers). Each question has ten options (A-J); exactly one is correct."
# The held-out demonstrations: KINA `index` values, in the order they are shown. Rule: heldout_shots.select_shots over all
# 899 `index` values with the gold letter as label (five distinct gold letters), unit-tested against the pinned dataset.
SHOT_INDEX = (876, 754, 328, 47, 820)


def _index_of(row: dict) -> int:
    return int(row["index"])


def kina_base_prompt(line, task_name: str = None):
    # A held-out question is a demonstration only: it is not an evaluation item.
    if not line.get("__few_shots") and int(line["index"]) in SHOT_INDEX:
        return None
    options = _option_texts(line)  # explicit option keys; the option explanations are never shown
    body = f"Question: {line['question'].strip()}\n"
    body += "\n".join(f"{label}: {option}" for label, option in zip(_LABELS, options))
    body += "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=f"{HEADER}\n\n{body}",
        choices=[f" {label}" for label in _LABELS],
        gold_index=_LABELS.index(line["correct_answer"].strip()),
        instruction=HEADER,
        specific={"index": int(line["index"]), "discipline": line["discipline"].split("/", 1)[0].strip()},
    )


def shot_rows() -> list[dict]:
    return rows_by_key(REPO, "default", "test", REVISION, _index_of, SHOT_INDEX)


TASKS_TABLE = [
    LightevalTaskConfig(
        name="kina_base:mcf_em",
        prompt_function=kina_base_prompt,
        hf_repo=REPO,
        hf_subset="default",
        hf_revision=REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_rows=shot_rows,
        few_shots_select="sequential",  # the first K of the five fixed demonstrations, in order
        generation_size=5,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=0,
    )
]
