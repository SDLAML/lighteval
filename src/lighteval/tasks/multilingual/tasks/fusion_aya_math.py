"""
name:
Fusion Aya Math Bench

dataset:
tiny-aya-math-edition/fusion-aya-math-bench

abstract:
Fusion Aya Math Bench is a multilingual, olympiad-level mathematical reasoning
dataset. Each problem is paired with a high-quality chain-of-thought solution
fused from the reasoning traces of different frontier models, derived from the
open-ended English subset of OlympiadBench.

Uses CoT-style prompts matching the mgsm.py setup ("Think step by step /
Problem: / Solution:"), with language-specific translations.

Reports expr_gold_metric (math expression / LaTeX \\boxed{} parser).

Version 1 (11-13 review fixes, titan_eval): (a) the stop strings only match at the START OF A LINE (a restarted problem), and the
colon-less aliases ("Problem", "Aufgabe", ...) only as a line of their own: version 0 cut solutions at the word inside a sentence or
heading ("1. **Understanding the Problem**", "Um die Aufgabe zu lösen": 9 of 129 English and 17 of 129 German outputs); (b) bare list /
tuple golds ("1,3,9,11", "(6,5)", "2400, 1350, 864") are written as $...$ so that gold and prediction go through the same LaTeX path:
a list is compared as a set (order does not matter), a tuple in order. In version 0 the bare list was read as the decimal 1.3 (or as its
last number: "2400, 1350, 864" -> 864, so a boxed 864 scored) and \\boxed{1,3,9,11} could never match.

languages:
english, german, french, ukrainian

tags:
math, multilingual, reasoning, olympiad, chain-of-thought

paper:
https://arxiv.org/abs/2510.00931
"""

import re

from langcodes import standardize_tag

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# Pinned dataset revision (tiny-aya-math-edition/fusion-aya-math-bench): the snapshot in the offline cache.
_REVISION = "56c020a524e50e03df8b0cf7d4efbf8ae85a9d9c"


_LANGUAGES = [
    Language.ENGLISH,
    Language.GERMAN,
    Language.FRENCH,
    Language.UKRAINIAN,
]


# Each entry is:
#   instruction, problem_label, solution_label
#
# The instruction means:
#   "Solve the following math problem. Think step by step before giving
#    the final answer."
_FUSION_AYA_PROMPT_PARTS: dict[Language, tuple[str, str, str]] = {
    Language.ENGLISH: (
        "Solve the following math problem. Think step by step before giving the final answer.",
        "Problem:",
        "Solution:",
    ),
    Language.GERMAN: (
        "Löse die folgende Mathematikaufgabe. Denke Schritt für Schritt nach, bevor du die endgültige Antwort gibst.",
        "Aufgabe:",
        "Lösung:",
    ),
    Language.FRENCH: (
        "Résous le problème de mathématiques suivant. Réfléchis étape par étape avant de donner la réponse finale.",
        "Problème:",
        "Solution:",
    ),
    Language.UKRAINIAN: (
        "Розв'яжи наступну математичну задачу. Міркуй покроково, перш ніж дати остаточну відповідь.",
        "Задача:",
        "Розв'язання:",
    ),
}


# Extra problem-label variants that models may generate (a restarted task written without the colon). Matched only as a LINE OF THEIR
# OWN ("\nProblem\n"): version 0 matched the bare word anywhere and cut "1. **Understanding the Problem**" and "Um die Aufgabe zu lösen".
# Keep this small: too many stop strings increase accidental truncation risk.
_FUSION_AYA_PROBLEM_LABEL_STOP_ALIASES: dict[Language, list[str]] = {
    Language.ENGLISH: ["Problem"],  # no-space variant
    Language.GERMAN: ["Aufgabe"],  # no-space variant
    Language.FRENCH: ["Problème"],  # no-space variant
    Language.UKRAINIAN: ["Задача"],  # no-space variant
}


def validate_fusion_aya_prompt_parts() -> None:
    """Fail fast if a language is missing localized prompt parts."""
    missing_languages = [
        language for language in _LANGUAGES if language not in _FUSION_AYA_PROMPT_PARTS
    ]

    if missing_languages:
        missing = ", ".join(language.value for language in missing_languages)
        raise ValueError(f"Missing Fusion Aya prompt parts for: {missing}")


def fusion_aya_cot_template(language: Language) -> str:
    """Return the localized Fusion Aya CoT template for a language."""
    try:
        instruction, problem_label, solution_label = _FUSION_AYA_PROMPT_PARTS[language]
    except KeyError as exc:
        raise ValueError(
            f"Missing Fusion Aya prompt parts for language: {language.value}"
        ) from exc

    return f"{instruction}\n\n{problem_label}\n{{prompt}}\n\n{solution_label}"


def fusion_aya_stop_sequences(language: Language) -> list[str]:
    """Return stop sequences for a language.

    Stop when the model appears to start a new problem: a line that starts with a problem label (version 1; version 0 matched the label
    anywhere in the text, so "Understanding the Problem:" in the model's own heading cut the solution).

    We keep English stops because models sometimes switch back to English even
    when the prompt is localized. We also add the current language's problem
    label and the colon-less aliases (as a line of their own) to avoid over-stopping.
    """
    try:
        _, problem_label, _ = _FUSION_AYA_PROMPT_PARTS[language]
    except KeyError as exc:
        raise ValueError(
            f"Missing Fusion Aya prompt parts for language: {language.value}"
        ) from exc

    stop_sequences = [
        "\nProblem:",
        f"\n{problem_label}",
        *(f"\n{alias}\n" for alias in _FUSION_AYA_PROBLEM_LABEL_STOP_ALIASES.get(language, [])),
    ]

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(stop_sequences))


_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def fusion_aya_gold(answer: str) -> str:
    """The gold in the form the scorer reads: a bare list / tuple is wrapped in $...$ (version 1).

    Gold answers are OlympiadBench strings. Integers, decimals ("4.25") and the golds already written in LaTeX ("$\\frac{4}{7}$",
    "$64,\\frac{1}{4}$") are left as they are. The 11 bare comma lists and tuples ("1,3,5", "2400, 1350, 864", "(6,5)",
    "(1,-4,-2),(3,2,3),(13,2,-2)"; 7 + 3 per language) are written as $...$: without it the extractor reads "1,3,5" as the decimal
    1.3 and "2400, 1350, 864" as 864 (a boxed 864 scored), and the prediction \\boxed{1,3,5} parsed as a set could never equal it.
    As LaTeX, a comma list is a set (the order of the answer does not matter) and a parenthesised list is a tuple (ordered).
    """
    text = str(answer).strip()
    if not text or "$" in text or "\\boxed" in text or _PLAIN_NUMBER.fullmatch(text):
        return answer
    if "," in text or text[0] in "([":
        return f"${text}$"
    return answer


def fusion_aya_cot_prompt(language: Language):
    """Build a CoT prompt function for Fusion Aya Math in the given language."""
    template = fusion_aya_cot_template(language)

    def prompt_fn(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=template.format(prompt=line["question"]),
            choices=[fusion_aya_gold(line["answer"])],
            gold_index=0,
        )

    return prompt_fn


validate_fusion_aya_prompt_parts()


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"fusion_aya_math:{language.value}:gen",
        prompt_function=fusion_aya_cot_prompt(language),
        hf_repo="tiny-aya-math-edition/fusion-aya-math-bench",
        hf_revision=_REVISION,
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("benchmark",),
        few_shots_split=None,
        generation_size=4096,
        metrics=[
            Metrics.expr_gold_metric,
        ],
        stop_sequence=fusion_aya_stop_sequences(language),
        version=1,  # 1: stops at the start of a line only; list / tuple golds written as $...$ (0: cut "Understanding the Problem"; lists never matched)
    )
    for language in _LANGUAGES
]
