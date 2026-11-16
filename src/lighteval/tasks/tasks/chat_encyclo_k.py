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
Encyclo-K (chat), the official protocol for post-trained models

dataset:
m-a-p/Encyclo-K

abstract:
Encyclo-K (arXiv 2512.24867, Liang et al.) composes 8-10 textbook knowledge statements into one multiple-choice question with 4-8 options, each option a
combination of statements; the model picks the option whose statements are all true. 5,038 questions (3,286 English / 1,752 Chinese by the paper's Table 2), 11
disciplines. This is the chat twin of the official evaluation code (github.com/multimodal-art-projection/Encyclo-K @ 5549c738): ALL 5,038 questions, one user
message = the benchmark's own zero-shot prompt (`config/prompt/zero-shot.yaml` filled by `infer/data_loader.py` l.16-18, no system prompt:
`utils/build_conversation.py`), the official answer extraction (`eval/eval.py` l.32-105 and l.131-136: last line first, then the whole text, three
patterns each, then a search for the option text), accuracy = correct / all questions (sample-wise, the paper's "Avg."), a question without an extracted answer
is wrong ("miss"). The scored text is what follows the LAST </think> (an unclosed block = no answer). Per-discipline accuracies are the columns of the paper's
Table 3. `acc_anchored` is a DIAGNOSTIC (the shared strict `Answer: X` reader of the other chat tasks). The generation budget and the sampling come from the run
(titan_eval registry yaml: 32,768 thinking / the official 4,096 non-thinking).

languages:
english, chinese

tags:
chat, knowledge, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2512.24867
"""

import logging
import re
from string import ascii_uppercase

import numpy as np

from lighteval.metrics.chat_metrics import ChatSamples, anchored_letter
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.chat_evals import BUDGET
from lighteval.tasks.tasks.encyclo_k_base import REPO, REVISION


try:
    import regex as _engine  # the official eval.py asks for it (l.16-21 TODO): `re` has no timeout and the official patterns can backtrack exponentially

    _TIMEOUT = True
except ImportError:  # pragma: no cover (llm_env has it: a dependency of math-verify)
    _engine = re
    _TIMEOUT = False

logger = logging.getLogger(__name__)

TASK_VERSION = 0
REGEX_TIMEOUT = 5            # seconds per search: the official `@timeout_decorator.timeout(5)` (eval.py l.14); there it cannot interrupt `re`, here `regex` does
DISCIPLINES = ["Agriculture", "Economics", "Education", "Engineering", "History", "Law", "Literature", "Management", "Medicine", "Philosophy", "Science"]
DISCIPLINE_KEYS = {d: f"acc_{d.lower()}" for d in DISCIPLINES}
NAMES = ("acc", "miss", "acc_anchored", *DISCIPLINE_KEYS.values())

# ------------------------------------------------------------------------------------------------------------------------------ prompt
# config/prompt/zero-shot.yaml (the `prompt_format` block: YAML `|` keeps ONE trailing newline), filled by infer/data_loader.py l.16-18:
#   options_format = ', '.join(chr(65+i) ...);  prompt_format = [question + '\n' + '\n'.join(f'{chr(65+i)}) {option}' ...)];  template['prompt_format'][0].format(options_format, *prompt_format)
# The dataset's own `prompt` column is the same text (checked for all 5,038 rows by the unit test).
PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. Please carefully analyze each statement and option, and end your response with "
    "'Answer: $OPTION_NUMBER' (no quotes), where OPTION_NUMBER is {}.\n\n{}\n"
)


def official_prompt(question: str, options: list[str]) -> str:
    options_format = ", ".join(chr(65 + i) for i in range(len(options)))
    body = question + "\n" + "\n".join(f"{chr(65 + i)}) {option}" for i, option in enumerate(options))
    return PROMPT_TEMPLATE.format(options_format, body)


_HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF\U00030000-\U0003134F]")


def language_of(question: str) -> str:
    """Informational only (`doc.specific["lang"]`, not scored): "zh" when at least 10% of the question's characters are Han characters. The dataset has no language field; this
    rule reproduces the paper's Table 2 English counts exactly (3,286 in total and in each of the 11 disciplines, checked 11-16)."""
    return "en" if len(_HAN_RE.findall(question)) / max(1, len(question)) < 0.10 else "zh"


def encyclo_k_chat_prompt(line, task_name: str = None):
    options = [str(o) for o in line["options"]]
    letters = list(ascii_uppercase[: len(options)])
    # gold = `answer_letter`, as eval.py l.179-181 compares it; a letter outside the option range raises (no silent default)
    return Doc(
        task_name=task_name,
        query=official_prompt(line["question"], options),     # no `instruction`: the prompt manager must not re-attach or strip anything
        choices=letters,
        gold_index=letters.index(line["answer_letter"]),
        specific={
            "discipline": line["discipline"],
            "field": line["field"],
            "subfield": line["subfield"],
            "difficulty": line["difficulty"],
            "lang": language_of(line["question"]),
            "options": options,                                  # the option-text fallback of the official extractor reads them
            "uuid": line["uuid"],
        },
    )


# ------------------------------------------------------------------------------------------------------------------- official extraction
# eval/eval.py l.32-105 @ 5549c738, the patterns evaluated as the file's (non-raw) f-strings give them, written here as raw strings with the option set as a placeholder; the
# unit test holds the evaluated official strings. Facts of the official reader that matter (all kept):
#  - the patterns are applied with re.IGNORECASE (letters): a lower-case letter is extracted too and is then WRONG (the comparison with `answer_letter` is case-sensitive), not a miss;
#  - the letters are always A-J (`extract_option_labels(.., 'ABCDEFGHIJ')`), whatever the option count; a letter beyond the options is simply wrong;
#  - the last line of the text is tried first (pattern 1, 2, 3), then the WHOLE text (pattern 1, 2, 3, the first match in the text, not the last);
#  - pattern 3 reads a bare letter at the start of a line ("A", "(B)", "C."), also an article ("A statement ...");
#  - a hedge ("Answer: B or C") gives its first letter.
_LABEL_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([@@])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)[\*\s]*:\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([@@])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"^[^\w\r\n]*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([@@])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
)
_CONTENT_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(@@)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(@@)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"^[^\w\r\n]*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(@@)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
)
OFFICIAL_LETTERS = "ABCDEFGHIJ"


def _search(pattern: str, text: str, flags: int = 0):
    """eval.py `safe_regex_search` (l.15-30): a timeout or an error gives no match (the official prints and returns None)."""
    try:
        if _TIMEOUT:
            return _engine.search(pattern, text, flags, timeout=REGEX_TIMEOUT)
        return _engine.search(pattern, text, flags)
    except TimeoutError:
        logger.warning("Encyclo-K answer regex timed out (%s s) on a text of %d characters: no match", REGEX_TIMEOUT, len(text))
    except Exception as error:  # noqa: BLE001  (the official returns None for every error)
        logger.warning("Encyclo-K answer regex error: %s", error)
    return None


def label_patterns(letters: str = OFFICIAL_LETTERS) -> tuple[str, ...]:
    return tuple(p.replace("@@", letters) for p in _LABEL_PATTERNS)


def content_patterns(options: list[str]) -> tuple[str, ...]:
    alternatives = "|".join(re.escape(o) for o in options)
    return tuple(p.replace("@@", alternatives) for p in _CONTENT_PATTERNS)


def extract_option_labels(text: str) -> str | None:
    """eval.py l.32-69 with `options='ABCDEFGHIJ'`: the letter of the first matching pattern, last line first, then the whole text."""
    text = text.rstrip()
    last_line = text.split("\n")[-1]
    patterns = label_patterns()
    for pattern in patterns:
        match = _search(pattern, last_line, _engine.IGNORECASE)
        if match:
            return match.group(1)
    for pattern in patterns:
        match = _search(pattern, text, _engine.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_option_content(text: str, options: list[str]) -> str | None:
    """eval.py l.71-105: the text of an option named in the answer (case-sensitive), last line first, then the whole text."""
    text = text.rstrip()
    last_line = text.split("\n")[-1]
    patterns = content_patterns(options)
    for pattern in patterns:
        match = _search(pattern, last_line)
        if match:
            return match.group(1)
    for pattern in patterns:
        match = _search(pattern, text)
        if match:
            return match.group(1)
    return None


def official_answer(text: str, options: list[str]) -> str | None:
    """eval.py l.131-136 (zero-shot branch): the label reader, else the option-text reader mapped to its letter. None = "miss"."""
    predict = extract_option_labels(text)
    if predict is None:
        content = extract_option_content(text, options)
        if content:
            if content not in options:  # cannot happen (the match is one of the alternatives); the official would raise
                logger.warning("Encyclo-K option-text match %r is not an option", content)
                return None
            predict = chr(options.index(content) + 65)
        else:
            predict = None
    return predict


# --------------------------------------------------------------------------------------------------------------------------------- metric
def _score(doc: Doc, pred: str) -> dict[str, float]:
    """One sample. acc = the official extraction equals the gold letter (eval.py l.181); miss = no answer extracted (l.187); acc_anchored = the shared strict reader (diagnostic);
    acc_<discipline> = acc for the questions of that discipline, NaN for the others (the corpus value is the mean over the members)."""
    spec = doc.specific or {}
    gold = doc.choices[doc.gold_index]
    out = {name: float("nan") for name in DISCIPLINE_KEYS.values()}
    if not pred or not pred.strip():
        acc, miss, strict = 0.0, 1.0, 0.0
    else:
        predict = official_answer(pred, spec["options"])
        acc, miss = float(predict == gold), float(predict is None)
        strict = float(anchored_letter(pred, len(doc.choices)) == gold)
    out.update({"acc": acc, "miss": miss, "acc_anchored": strict})
    out[DISCIPLINE_KEYS[spec["discipline"]]] = acc
    return out


def _members_avg(values) -> float:
    """Mean over the members of a discipline (NaN = not a member). No "mean" in the name on purpose: lighteval gives "mean" aggregations the plain stderr, which NaN would poison,
    and bootstraps every other aggregation (as the LongBench v2 breakdowns)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def encyclo_k_chat_metric(n: int = 1) -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
        metric_name=list(NAMES),
        sample_level_fn=ChatSamples(n, _score, NAMES, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: (np.mean if name in ("acc", "miss", "acc_anchored") else _members_avg) for name in NAMES},
        higher_is_better={name: name != "miss" for name in NAMES},
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="encyclo_k_chat",
        prompt_function=encyclo_k_chat_prompt,
        hf_repo=REPO,
        hf_subset="default",
        hf_revision=REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[encyclo_k_chat_metric(1)],
        version=TASK_VERSION,
    )
]
