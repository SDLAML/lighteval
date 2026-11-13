"""
name:
Encyclo-K (English), base-model protocol (capped chain of thought)

dataset:
m-a-p/Encyclo-K

abstract:
Encyclo-K questions list 8-10 knowledge statements and ask which combination is true / false, with 4-8 lettered options; the
benchmark's own prompt asks the model to "carefully analyze each statement and option" and to end with "Answer: <letter>", so the
task is long-form reasoning by design (a direct letter after the question is near chance for small models and is not offered).
This variant is the capped version for base-model checkpoints: 0-shot, the benchmark's prompt unchanged, at most 4096 new tokens
(Oleg Filatov's encyclo_k:* allow 32768), generation stopped when the model starts a new "Answer the following multiple choice
question" prompt, and a STRICT answer reader: the answer is the FIRST line that reads exactly "Answer: X" (case-insensitive,
optional markdown bold / parentheses / dollar signs / trailing dot) with X one of the question's option letters. A letter anywhere
else in the text earns nothing (gpqa_instruct_metric accepts an unanchored letter). Version 1 (owner OK 11-13) also accepts the two
unambiguous forms a base model writes for this prompt, "Answer: B) <option text>" (the letter, a closing parenthesis, then the option
copied; not a hedge such as "B) or C)") and the repeated prefix "Answer: Answer: B" (also "Answer: Answer: B) <text>"): version 0
rejected 72.5% of Qwen3-1.7B-Base's outputs for them, so its score measured the format, not the knowledge. Everything else stays
strict. Metrics: extractive_match and answered (share of generations with a valid answer line).
English subset = questions with NO Han character in the question text or the options (3,256 of 5,038). Oleg Filatov's
encyclo_k_en used the file_name instead, which wrongly keeps 20 questions with Chinese statements (English book title) and drops
408 English questions of Chinese-titled books ('有机化学', '量子力学'). One pooled task over the 11 disciplines (the discipline, field
and difficulty are kept in doc.specific for a breakdown from the details).

languages:
english

tags:
general-knowledge, knowledge, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2512.24867
"""

import re
from string import ascii_uppercase
from typing import Any

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.encyclo_k import _HAN_RE


REPO = "m-a-p/Encyclo-K"
REVISION = "0f81da9652b9c4d7bad73ce9576d81901d9441f2"  # main on 2026-11-13 (card: license MIT)
STOP_SEQUENCES = ["Answer the following multiple choice question"]
TASK_VERSION = 1  # 1: also "Answer: B) <option text>" and "Answer: Answer: B" (0: only the bare answer line)
# "Answer: B", "ANSWER: (B)", "**Answer:** B.", "Answer: $B$" on a line of its own; since version 1 also "Answer: Answer: B" (the prompt's own
# instruction "end your response with 'Answer: $OPTION_NUMBER'" copied once more) and "Answer: B) <option text>" (the letter, ")", then the
# option's text, which is not a hedge ("B) or C)") and holds no second option marker ("C)"); with or without the repeated prefix).
# Nothing else counts. The letter itself must be upper case (a lower-case "a" is an article).
# The pieces are written so that no two neighbouring quantifiers can split the same run of blanks (`[ \t*_]*` is ONE class): a generation of
# thousands of spaces after "Answer:" must not make the search quadratic or worse (the first draft with `[ \t]*[*_]*[ \t]*` took 8 s for 2,000 blanks).
_PREFIX = r"[ \t>#*_]*answer[ \t]*:[ \t*_]*"
ANSWER_LINE = re.compile(
    rf"^{_PREFIX}(?:answer[ \t]*:[ \t*_]*)?"  # "Answer:" and optionally the instruction's text once more
    r"(?:"
    r"\$?\(?((?-i:[A-Z]))\)?\$?[ \t*_]*(?:\.[ \t]*)?"  # the strict form: the letter alone on the line
    r"|((?-i:[A-Z]))\)[ \t]+(?=\S)(?!(?:or|and|und|oder|ou|et)\b|[/,&])(?![^\n]*(?<![A-Za-z0-9])(?-i:[A-Z])\))[^\n]*"  # "B) <option text>"
    r")$",
    re.IGNORECASE | re.MULTILINE,
)


def is_english(line) -> bool:
    return _HAN_RE.search(line["question"]) is None and not any(_HAN_RE.search(option) for option in line["options"])


def extract_answer(text: str, n_options: int) -> str | None:
    """The letter of the FIRST answer line, if it is one of the first `n_options` letters; else None."""
    match = ANSWER_LINE.search(text or "")
    if match is None:
        return None
    letter = match.group(1) or match.group(2)
    return letter if letter in ascii_uppercase[:n_options] else None


def encyclo_k_base_prompt(line, task_name: str = None):
    if not is_english(line):
        return None
    letters = list(ascii_uppercase[: len(line["options"])])
    return Doc(
        task_name=task_name,
        query=line["prompt"],  # the benchmark's own prompt (instruction + statements + lettered options)
        choices=letters,
        gold_index=letters.index(line["answer_letter"]),
        specific={
            "discipline": line["discipline"],
            "field": line["field"],
            "difficulty": line["difficulty"],
            "is_calculation": line["is_calculation"],
            "n_options": len(letters),
        },
    )


class StrictAnswerLine(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> dict[str, Any]:
        text = (model_response.final_text[0] if model_response.final_text else "") or ""
        predicted = extract_answer(text, len(doc.choices))
        return {
            "extractive_match": float(predicted is not None and predicted == doc.choices[doc.gold_index]),
            "answered": float(predicted is not None),
        }


STRICT_ANSWER_METRIC = SampleLevelMetricGrouping(
    metric_name=["extractive_match", "answered"],
    sample_level_fn=StrictAnswerLine(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={"extractive_match": np.mean, "answered": np.mean},
    higher_is_better={"extractive_match": True, "answered": True},
)

TASKS_TABLE = [
    LightevalTaskConfig(
        name="encyclo_k_base:cot",
        prompt_function=encyclo_k_base_prompt,
        hf_repo=REPO,
        hf_subset="default",
        hf_revision=REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=4096,
        metrics=[STRICT_ANSWER_METRIC],
        stop_sequence=STOP_SEQUENCES,
        version=TASK_VERSION,
    )
]
