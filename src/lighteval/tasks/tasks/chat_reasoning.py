# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
name:
Chat reasoning / knowledge (titan_eval, post-trained models): MMLU (chat), GPQA main (chat)

dataset:
lighteval/mmlu (the revision of the base task mmlu:mcf_em; the same questions as cais/mmlu test, 14,042), Idavidrein/gpqa gpqa_main (448, gated)

abstract:
Chat twins of the knowledge multiple-choice tasks, built exactly like the chat tasks of chat_evals.py (0-shot, the model's chat template,
the instruction "Answer: $LETTER" with "Think step by step before answering", the scored answer is the anchored "Answer: X" in the text
after the LAST </think>). Subtasks: mmlu_chat:<subject> (57 subjects, macro mean). Metric: chat_metrics.chat_choice_metric.
GPQA main uses the GPQA-Diamond prompt and the md5 option order of chat_evals.gpqa_chat_prompt, n = 8 samples per question.

languages:
english

tags:
chat, knowledge, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2009.03300 (MMLU), https://arxiv.org/abs/2311.12022 (GPQA)
"""

import re
from functools import partial
from string import ascii_uppercase

from lighteval.metrics.chat_metrics import _ANSWER, _HEDGE, _grouping, chat_choice_metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.agieval import _REVISION as _AGIEVAL_REVISION, AGIEVAL_SUBSETS, _is_valid_sample, record_to_sample
from lighteval.tasks.tasks.bigbench_hard import _BBH_SUBSETS
from lighteval.tasks.tasks.chat_evals import BUDGET, N_GPQA, gpqa_chat_prompt, mc_query
from lighteval.tasks.tasks.mmlu import _MMLU_REVISION, _MMLU_SUBSETS


# lukaemon/bbh @ 982bb89f (the BBH test split, 27 subsets; the base task bigbench_hard has no pin)
_BBH_REVISION = "982bb89fd79532a8ac676a61fc42eb1aeec63f99"


# ----------------------------------------------------------------------------------------------------------- MMLU
def mmlu_chat_prompt(line, task_name: str = None, *, subject: str):
    """The question and its 4 options, the chat_evals instruction (letters of the question); gold = the answer index (lighteval/mmlu: int)."""
    options = list(line["choices"])
    gold = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else int(line["answer"])
    return Doc(
        task_name=task_name,
        query=mc_query(line["question"], options),
        choices=list(ascii_uppercase[: len(options)]),
        gold_index=gold,
        specific={"subject": subject},
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu_chat:{subset}",
        prompt_function=partial(mmlu_chat_prompt, subject=subset),
        hf_repo="lighteval/mmlu",
        hf_subset=subset,
        hf_revision=_MMLU_REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[chat_choice_metric(1)],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# ------------------------------------------------------------------------------------------------------ GPQA main
TASKS_TABLE += [
    LightevalTaskConfig(
        name="gpqa_main_chat",
        prompt_function=gpqa_chat_prompt,
        hf_repo="Idavidrein/gpqa",
        hf_subset="gpqa_main",
        hf_revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[chat_choice_metric(N_GPQA)],
        version=0,
    ),
]


# ------------------------------------------------------------------------------------------------------------ AGIEval
def agieval_en_chat_prompt(line, task_name: str = None):
    """The English AGIEval multiple-choice item (the base task's record_to_sample: passage + question, options, label) as a chat_evals MC prompt.
    Rows the base task filters out (_is_valid_sample: two identical SAT-English rows) return None."""
    sample = record_to_sample(line)
    if not _is_valid_sample(sample):
        return None
    labels = ascii_uppercase[: len(sample.choices)]
    return Doc(
        task_name=task_name,
        query=mc_query(sample.input, list(sample.choices)),
        choices=list(labels),
        gold_index=labels.index(sample.target),
    )


TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"agieval_en_chat:{subset}",
        prompt_function=agieval_en_chat_prompt,
        hf_repo="lighteval/agi_eval_en",
        hf_subset=subset,
        hf_revision=_AGIEVAL_REVISION,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[chat_choice_metric(1)],
        version=0,
    )
    for subset in AGIEVAL_SUBSETS
]


# ---------------------------------------------------------------------------------------------------------- BBH CoT
# Answer types. "letter": the target is "(X)" and the letter is scored (anchored, A-R: reasoning_about_colored_objects has up to 18 options).
# "text": the target is the literal answer; the scored text is what follows the last "Answer:" on that line, normalised (case, spaces, bold, trailing period).
_BBH_TEXT_FORMAT = {
    "boolean_expressions": "True or False",
    "causal_judgement": "Yes or No",
    "dyck_languages": "only the sequence of the closing brackets, separated by spaces",
    "formal_fallacies": "valid or invalid",
    "multistep_arithmetic_two": "the integer",
    "navigate": "Yes or No",
    "object_counting": "the integer",
    "sports_understanding": "yes or no",
    "web_of_lies": "Yes or No",
    "word_sorting": "the words in alphabetical order, separated by spaces",
}
_BBH_LETTERS = ascii_uppercase[:18]                       # A-R
_ANSWER_R = re.compile(_ANSWER.pattern.replace("A-J", "A-R"))
_HEDGE_R = re.compile(_HEDGE.pattern.replace("A-J", "A-R"))
_TEXT_ANSWER = re.compile(r"(?i)\banswer\b[ \t]*(?:is)?[ \t]*[:\uff1a][ \t]*([^\n]*)")


def _bbh_letter(pred: str, n_options: int) -> str | None:
    """anchored_letter of chat_metrics with the letters A-R (the same rules: hedges and out-of-range letters give None)."""
    last = None
    for m in _ANSWER_R.finditer(pred):
        last = None if _HEDGE_R.match(pred, m.end()) else m.group(1)
    if last is None or _BBH_LETTERS.index(last) >= n_options:
        return None
    return last


def _norm_text(s: str) -> str:
    s = s.replace("**", "").replace("`", "").strip().strip("$").strip()
    s = re.sub(r"\s+", " ", s).rstrip(".").strip()
    return s.lower()


def _bbh_score(doc: Doc, pred: str) -> dict[str, float]:
    """em = 1 when the anchored answer equals the gold of the subset's answer type (letter or normalised text); empty text = 0."""
    if not pred or not pred.strip():
        return {"em": 0.0}
    gold = doc.choices[0].strip()
    if re.fullmatch(r"\([A-R]\)", gold):                    # letter subset
        letter = _bbh_letter(pred, len(doc.specific["n_options"]))
        return {"em": 1.0 if letter is not None and f"({letter})" == gold else 0.0}
    found = _TEXT_ANSWER.findall(pred)
    if not found:
        return {"em": 0.0}
    return {"em": 1.0 if _norm_text(found[-1]) == _norm_text(gold) else 0.0}


def bbh_chat_prompt(line, task_name: str = None, *, subset: str):
    if subset in _BBH_TEXT_FORMAT:
        fmt = _BBH_TEXT_FORMAT[subset]
        n_options = 0
    else:
        n_options = len(re.findall(r"(?m)^\(([A-R])\)", line["input"])) or 2
        fmt = "the letter of the correct option in parentheses, e.g. (A)"
    instruction = (
        "Answer the following question. Think step by step before answering. The last line of your response should be of the following format: "
        f"'Answer: $ANSWER' (without quotes) where $ANSWER is {fmt}."
    )
    return Doc(
        task_name=task_name,
        query=f"{instruction}\n\n{line['input'].strip()}",
        choices=[line["target"].strip()],
        gold_index=0,
        specific={"subset": subset, "n_options": list(_BBH_LETTERS[:n_options])},
    )


TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"bbh_cot_chat:{subset}",
        prompt_function=partial(bbh_chat_prompt, subset=subset),
        hf_repo="lukaemon/bbh",
        hf_subset=subset,
        hf_revision=_BBH_REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[_grouping(1, _bbh_score, ("em",))],
        version=0,
    )
    for subset in _BBH_SUBSETS
]
