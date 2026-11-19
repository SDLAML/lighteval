"""titan_eval (owner 11-17): the scored answer of a BASE math generation task ends where the model starts to INVENT a new problem.

A base model that is not stopped goes on after its answer: "\\n\\nProblem:\\nA 100 piece puzzle ...", MGSM "\\nВопрос:", "\\n**প্রশ্ন 2:", "\\n### 问题 2：", "\\nPregunta:". The extractive-match scorers read the LAST
(or any later) number of the text, so they read the invented problem's numbers. The rule (analysed on the stored answers of every base model, tmp/p3/cutfix/analyse.py, RULES["broad"]):

    the scored answer = the generation up to the FIRST match of
        \\n [ \\t # * > -]*  (N. | N))?  (**)?  (<label> | 第N题)  \\s* digits? \\s* (**)?  \\s*  ( : | ：)         (re.IGNORECASE)

i.e. a LINE (the match starts at a line break) that begins, after markdown marks (# * > -), a list number and a bold mark, with a problem / question label and ends with a colon. One broad label set serves every task
(no per-language sets): the question word of every language of TRANSLATION_LITERALS, the problem labels of the MGSM and Fusion Aya Math prompts, and Problem, Question, Q, Exercise, Example, 题目, 問題, 问题; plus 第N题.
A label that is not at the start of a line (the model's own sentence "Understanding the Problem: ...", "the Question: ...") never cuts. A generation that starts with the label ("\\nProblem: ...") is cut to the
empty string, which scores 0 (no answer left).

`CutInventedProblem(metric)` = the existing sample-level metric computed on the CUT text (everything else, the extraction and the comparison, is the wrapped metric's own code). `Metrics.expr_gold_cut_metric` and
`Metrics.expr_gold_boxed_cut_metric` are `expr_gold_metric` / `expr_gold_boxed_metric` through it, under the SAME metric names (`extractive_match`, `extractive_match_boxed`), used by the base math tasks only (gsm8k,
gsm_plus, gsm_symbolic, math_500, math:gen, mgsm, fusion_aya_math); `Metrics.expr_gold_metric` itself is unchanged (other tasks use it).

11-19 (extraction audit of the base math tasks, same wrapper): (a) native digits are read as ASCII digits (`ascii_digits`), (b) a bare LaTeX variable followed on its line by a number / expression is not an answer candidate
(`skip_lone_variable`, extractive_match_utils.is_lone_variable_before_answer). Both only add credit that the answer text already shows; every other rule of the scorer is unchanged.

The label list is built from the sources on FIRST USE and cached: the two task modules (mgsm, fusion_aya_math) import `Metrics`, so this module cannot import them at its own import time
(a circular import); by the first scored item every module is loaded."""

import functools
import re
import unicodedata
from dataclasses import replace

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


# Labels that are in no prompt of the fork but start an invented problem in the stored answers of the base models.
EXTRA_LABELS = ("Problem", "Question", "Q", "Exercise", "Example", "题目", "問題", "问题")


def _bare(label: str) -> str:
    """A prompt label without its colon ("Problem:" -> "Problem"; the regex has its own colon part)."""
    return label.strip().rstrip(":：").rstrip()


@functools.lru_cache(maxsize=1)
def invented_problem_labels() -> tuple[str, ...]:
    """The broad label set, sorted: the question word of every language of TRANSLATION_LITERALS (a language without one is skipped), the problem labels of mgsm._MGSM_PROMPT_PARTS and
    fusion_aya_math._FUSION_AYA_PROMPT_PARTS (colon removed), and EXTRA_LABELS. Built from the sources, not from a file."""
    from lighteval.tasks.multilingual.tasks.fusion_aya_math import _FUSION_AYA_PROMPT_PARTS
    from lighteval.tasks.multilingual.tasks.mgsm import _MGSM_PROMPT_PARTS
    from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS

    labels = set(EXTRA_LABELS)
    for literals in TRANSLATION_LITERALS.values():
        try:
            word = literals.question_word
        except AttributeError:      # TranslationLiterals raises for a word that is not translated yet
            continue
        labels.add(_bare(word))
    for parts in (_MGSM_PROMPT_PARTS, _FUSION_AYA_PROMPT_PARTS):
        for _instruction, problem_label, _solution_label in parts.values():
            labels.add(_bare(problem_label))
    labels.discard("")
    return tuple(sorted(labels))


def invented_problem_regex(labels) -> "re.Pattern[str]":
    """The cut rule for a label set (module doc). Longest label first, as the analysis built it."""
    alternatives = "|".join(sorted((re.escape(label) for label in labels), key=len, reverse=True))
    return re.compile(
        r"\n[ \t#*>\-]*(?:\d+[.)]\s*)?(?:\*\*)?(?:" + alternatives + r"|第\s*\d+\s*题)\s*\d*\s*(?:\*\*)?\s*[:：]",
        re.IGNORECASE,
    )


@functools.lru_cache(maxsize=1)
def invented_problem_re() -> "re.Pattern[str]":
    """The compiled rule of the broad label set."""
    return invented_problem_regex(invented_problem_labels())


def cut_position(text: str) -> int | None:
    """Where the first invented problem starts (the index of its line break), None = no cut."""
    match = invented_problem_re().search(text)
    return match.start() if match else None


def cut_invented_problem(text: str) -> str:
    """`text` up to the first invented-problem line (the text itself when there is none)."""
    if not text:
        return text
    position = cut_position(text)
    return text if position is None else text[:position]


# titan_eval 11-19: the number extractors read ASCII digits only (a sympy-eval safety rule of extractive_match_utils), so an answer written in the digits of its language
# ("উত্তর: ২৫", "उत्तर: २५", "الجواب: ٢٥", Burmese, Khmer, fullwidth "２５") never was a number and scored 0 (MGSM, 40 languages: Qwen3.5 117 items, gemma 78, Nemotron 17). The golds are ASCII.
_ARABIC_SEPARATORS = {0x066B: ".", 0x066C: ","}      # ARABIC DECIMAL SEPARATOR, ARABIC THOUSANDS SEPARATOR


def ascii_digits(text: str) -> str:
    """`text` with every non-ASCII decimal digit (Unicode category Nd: Bengali, Devanagari, Arabic-Indic, Persian, Khmer, Burmese, Thai, fullwidth ...) replaced by its ASCII digit, and the Arabic decimal /
    thousands separator by "." / ",". Superscripts, fractions and other numerals (category No / Nl) are left alone; an ASCII text is returned as it is."""
    if text.isascii():
        return text
    return "".join(str(unicodedata.decimal(c)) if (not c.isascii() and unicodedata.category(c) == "Nd")
                   else _ARABIC_SEPARATORS.get(ord(c), c) for c in text)


class CutInventedProblem(SampleLevelComputation):
    """`metric` (a sample-level extractive-match computation) on the generation cut at the first invented problem, with native digits read as ASCII digits (`ascii_digits`). The cut text goes into `text` (and
    `text_post_processed` when set), so the wrapped metric reads exactly what it would read from a generation that stopped there; `doc.specific` gets the extraction of the cut text.
    The wrapped scorer is a private copy (metrics.py passes a deepcopy) and gets `skip_lone_variable = True` (11-19: a bare LaTeX variable followed by a number on its line is not an answer, see
    extractive_match_utils.is_lone_variable_before_answer): the rules the base math tasks share (cut, digits, lone variable) live in this one wrapper."""

    def __init__(self, metric: SampleLevelComputation):
        self.metric = metric
        self.metric.skip_lone_variable = True

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        cut = replace(
            model_response,
            text=[ascii_digits(cut_invented_problem(t)) for t in model_response.text],
            text_post_processed=(
                None
                if model_response.text_post_processed is None
                else [ascii_digits(cut_invented_problem(t)) for t in model_response.text_post_processed]
            ),
        )
        return self.metric.compute(doc, cut, **kwargs)
