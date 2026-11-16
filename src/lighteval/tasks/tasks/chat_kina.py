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
KINA (chat), the paper's / official protocol for post-trained models

dataset:
2077AIDataFoundation/KINA

abstract:
KINA (Knowledge Index of Noah's Ark, arXiv 2606.05104, Jin et al.) has 899 graduate-level knowledge questions with ten options (A-J), 12 top-level disciplines.
This is the chat twin of the official evaluation (github.com/2077AI/KINA @ 69a97f30, `src/kina_bench/utils.py` + `run_openai_chat.py`; paper section 5 and
appendix F.2): ALL 899 questions, ONE user message = the official prompt (`prompt_template`, utils.py l.110-114: "Answer the following multiple choice question ...
Think step by step before answering.\\nQuestion:\\n{question}\\nOptions:\\n{A: ...}"; the same text as the paper's appendix F.2 and as `_COT_INSTRUCTION` of Oleg
Filatov's `kina.py`), no system prompt, the question text as the dataset has it (the official code does not strip it), the answers stripped. The answer is read
with the OFFICIAL reader, which is lighteval 0.9.2's `gpqa_instruct_pass_at_1_*n` metric (utils.py l.15-27, l.129-177; pinned `lighteval==0.9.2`, pyproject.toml l.12):
`multilingual_extractive_match_metric(language=ENGLISH, IndicesExtractionConfig("NativeLetters"))` WITH its default `try_extract_without_anchor=True` (ported here as
`official_letter`; Filatov's `kina:*:cot` uses the newer fork reader with `try_extract_without_anchor=False`, a different reader). The paper reports avg@4
(section 5: "avg@4 accuracy at default temperature, with 32K maximum new tokens"): n = 4 samples per question, accuracy = correct / all 899 questions (the mean over the
samples of a question, then over the questions), a question without a letter in the answer is wrong. The scored text is what follows the LAST </think> (an unclosed
block = no answer). Per-discipline accuracies are the 12 discipline columns of the paper's Table 3. `acc_anchored` is a DIAGNOSTIC (the shared strict `Answer: X`
reader of the other chat tasks). The generation budget and the sampling come from the run (titan_eval registry yaml: 32,768, the paper's 32K).

languages:
english

tags:
chat, knowledge, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2606.05104
"""

import re

import numpy as np

from lighteval.metrics.chat_metrics import ChatSamples, anchored_letter
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.chat_evals import BUDGET
from lighteval.tasks.tasks.kina import _COT_INSTRUCTION, _KINA_DISCIPLINES, _LABELS, _option_texts
from lighteval.tasks.tasks.kina_base import REPO, REVISION


TASK_VERSION = 0
N_SAMPLES = 4        # the paper's avg@4 (section 5: "All results are reported as avg@4 accuracy"); the official CLI supports 1, 4 or 8 (utils.py KINA_METRICS)
OFFICIAL_COMMIT = "69a97f305af11d0001854d3392b8410f92fbaae3"   # github.com/2077AI/KINA main on 2026-11-16 (last commit 2026-05-28); the files below are read at this commit
DISCIPLINES = list(_KINA_DISCIPLINES)           # the 12 top-level disciplines in the order of the paper's Table 3 columns (alphabetical)
DISCIPLINE_KEYS = {d: "acc_" + d.lower().replace(" ", "_") for d in DISCIPLINES}
NAMES = ("acc", "miss", "acc_anchored", *DISCIPLINE_KEYS.values())

# ------------------------------------------------------------------------------------------------------------------------------ prompt
# utils.py l.110-114 (`prompt_template`) and l.117-126 (`get_messages`): ONE user message, no system prompt; `question` is `item["question"]` unchanged (load_data l.87), the
# options are the keys sorted (A..J), each `"{letter}: {answer.strip()}"` (load_data l.39 strips the answer), joined by "\n". The task builds it from Filatov's `_COT_INSTRUCTION`
# (kina.py l.55-59, equal to the first line of the official template; unit-tested against the official text) and `_option_texts` (the explicit key order, answers stripped).
PROMPT_TEMPLATE = (     # the official template, verbatim (utils.py l.110-114); `official_prompt` builds the same text from kina.py's pieces, the unit test holds the two equal on all 899 items
    "Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), "
    "where LETTER is one of A, B, C, D, E, F, G, H, I, or J. Think step by step before answering.\n"
    "Question:\n{question}\nOptions:\n{options}"
)


def official_prompt(question: str, options: list[str]) -> str:
    """The text of the official `get_messages` for one item: `options` = the ten answer texts in key order A-J."""
    option_str = "\n".join(f"{letter}: {answer.strip()}" for letter, answer in zip(_LABELS, options))
    return _COT_INSTRUCTION + "\nQuestion:\n" + question + "\nOptions:\n" + option_str


def discipline_of(path: str) -> str:
    """The root of the hierarchical discipline path ("Engineering/Civil Engineering/..." -> "Engineering"), as kina.py's subtasks and the paper's Table 3."""
    return path.split("/", 1)[0].strip()


def kina_chat_prompt(line, task_name: str = None):
    gold = line["correct_answer"].strip().upper()           # utils.py l.78
    if len(gold) != 1 or gold not in _LABELS:               # `"" in "ABC"` and `"AB" in "ABC"` are True for a str: test the length too (reward_checks/multiple_choice.md, code trap)
        raise ValueError(f"KINA item {line.get('index')}: gold {line['correct_answer']!r} is not one of {_LABELS}")
    return Doc(
        task_name=task_name,
        query=official_prompt(line["question"], _option_texts(line)),   # no `instruction`: the prompt manager must not re-attach or strip anything
        choices=list(_LABELS),
        gold_index=_LABELS.index(gold),
        specific={"index": int(line["index"]), "discipline": discipline_of(line["discipline"]), "path": line["discipline"]},
    )


# ------------------------------------------------------------------------------------------------------------------- official reader
# lighteval 0.9.2 (the version `kina_bench` pins; wheel lighteval-0.9.2-py3-none-any.whl, sha256 24905ae8..., read from the PyPI file, not executed) `lazy_indices_regex`
# (metrics/utils/extractive_match_utils.py l.283-337) for ENGLISH, `IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)` (the dataclass default
# of 0.9.2, l.83-92; the current fork's default is False), 10 choices -> the letters A-J; the pattern strings are those of the 0.9.2 source with its f-strings evaluated (literals of ENGLISH: full stop ".", comma ",", colon ":", space " ", answer word "answer"); compiled WITHOUT flags (the case-insensitive parts are the
# inline `(?i:...)` groups), `.` does not match a newline.
# `extract_target_from_pred(.., fallback_mode="first_match", extraction_mode="any_match")` (0.9.2, l.516-585, called by dynamic_metrics.py l.188-280): the patterns are tried by ascending priority; the first priority with a match decides;
# among its matches the one that ENDS last (ties: starts first) is taken. A letter is always extracted (`extract_match` for indices cannot fail), so the answer is that match's letter.
# The gold is extracted from the bare letter ("C") by the last pattern (priority 300) and compared as a stripped string.
# Facts of the official reader that matter (all kept): the unanchored patterns (priority 250 / 300) read ANY capital A-J of the text (the rightmost one), also the "A" of an article or the
# "I" of a pronoun; the anchored ones (0-200) need the letter followed by a full stop, comma, colon, space, `**` or the end ("Answer: B)" and "Answer: (B)" are NOT anchored matches);
# lower-case letters never match; a hedge "Answer: B or C" gives B; with two answers the LAST one counts; a text without any capital A-J letter has no answer (the official code raises
# `max([])` there: nothing to score; here it is a miss and wrong).
_LETTERS = "ABCDEFGHIJ"
_FULL_STOP, _COMMA, _COLON, _SPACE = ".", ",", ":", " "
_INDICE = f"(?P<indices>{'|'.join([re.escape(i) for i in _LETTERS])})"
_FULL_STOP_RE = rf"[{re.escape(_FULL_STOP)}\.]"
_COMMA_RE = rf"[{re.escape(_COMMA)}\,]"
_COLON_RE = rf"[{re.escape(_COLON)}\:]"
_SPACE_RE = re.escape(_SPACE)
_ANSWER_PREFIX_RE = rf"(^|{_SPACE_RE})(?:\*\*)?"
_ANSWER_SUFFIX_RE = rf"(?:\*\*)?(?:{_FULL_STOP_RE}|{_COMMA_RE}|{_COLON_RE}|{_SPACE_RE}|$)"
_ANSWER_RE = f"{_ANSWER_PREFIX_RE}{_INDICE}{_ANSWER_SUFFIX_RE}"
_ANSWER_RE_START = rf"^(?:\*\*)?{_INDICE}{_ANSWER_SUFFIX_RE}"
_ANSWER_WORD = "(?i:answer)"
OFFICIAL_PATTERNS = (               # (pattern, priority), ascending priority = the order they are tried
    (rf"(?i:final answer is)\:?\s*{_INDICE}\.?\s?I hope", 0),
    (rf"(?i:final answer.{{0,100}}?)\s+is\:?\s*{_INDICE}", 50),
    (f"{_ANSWER_WORD}{_COLON_RE}.{{0,50}}?{_ANSWER_RE}", 100),
    (f"{_ANSWER_WORD}.{{0,50}}?{_ANSWER_RE}", 150),
    (_ANSWER_RE_START, 200),
    (_ANSWER_RE, 250),
    (_INDICE, 300),
)
_COMPILED = tuple((re.compile(pattern), priority) for pattern, priority in OFFICIAL_PATTERNS)


def official_letter(text: str) -> str | None:
    """The letter the official reader extracts from `text` (a letter of A-J), None when the text holds no capital A-J at all (the official code has no value there)."""
    for pattern, _ in _COMPILED:
        matches = list(pattern.finditer(text))
        if matches:
            return max(matches, key=lambda m: (m.end(), -m.start())).group("indices")
    return None


# --------------------------------------------------------------------------------------------------------------------------------- metric
def _score(doc: Doc, pred: str) -> dict[str, float]:
    """One sample. acc = the official reader's letter equals the gold letter; miss = the official reader found no letter (an empty text, an unclosed think block); acc_anchored = the
    shared strict reader (diagnostic); acc_<discipline> = acc for the questions of that discipline, NaN for the others (the corpus value is the mean over the members)."""
    spec = doc.specific or {}
    gold = doc.choices[doc.gold_index]
    out = {name: float("nan") for name in DISCIPLINE_KEYS.values()}
    if not pred or not pred.strip():
        acc, miss, strict = 0.0, 1.0, 0.0
    else:
        letter = official_letter(pred)
        acc, miss = float(letter == gold), float(letter is None)
        strict = float(anchored_letter(pred, len(doc.choices)) == gold)
    out.update({"acc": acc, "miss": miss, "acc_anchored": strict})
    out[DISCIPLINE_KEYS[spec["discipline"]]] = acc
    return out


def _members_avg(values) -> float:
    """Mean over the members of a discipline (NaN = not a member). No "mean" in the name on purpose: lighteval gives "mean" aggregations the plain stderr, which NaN would poison."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def kina_chat_metric(n: int = N_SAMPLES) -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
        metric_name=list(NAMES),
        sample_level_fn=ChatSamples(n, _score, NAMES, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: (np.mean if name in ("acc", "miss", "acc_anchored") else _members_avg) for name in NAMES},
        higher_is_better={name: name != "miss" for name in NAMES},
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="kina_chat",
        prompt_function=kina_chat_prompt,
        hf_repo=REPO,
        hf_subset="default",
        hf_revision=REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[kina_chat_metric(N_SAMPLES)],
        version=TASK_VERSION,
    )
]
