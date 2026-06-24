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

languages:
english, german, french, ukrainian

tags:
math, multilingual, reasoning, olympiad, chain-of-thought

paper:
https://arxiv.org/abs/2510.00931
"""

from langcodes import standardize_tag

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


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


# Extra problem-label variants that models may generate.
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

    Stop when the model appears to start a new problem.

    We keep English stops because models sometimes switch back to English even
    when the prompt is localized. We also add the current language's problem
    label and a few high-value aliases to avoid over-stopping.
    """
    try:
        _, problem_label, _ = _FUSION_AYA_PROMPT_PARTS[language]
    except KeyError as exc:
        raise ValueError(
            f"Missing Fusion Aya prompt parts for language: {language.value}"
        ) from exc

    stop_sequences = [
        "Problem:",
        problem_label,
        *_FUSION_AYA_PROBLEM_LABEL_STOP_ALIASES.get(language, []),
    ]

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(stop_sequences))


def fusion_aya_cot_prompt(language: Language):
    """Build a CoT prompt function for Fusion Aya Math in the given language."""
    template = fusion_aya_cot_template(language)

    def prompt_fn(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=template.format(prompt=line["question"]),
            choices=[line["answer"]],
            gold_index=0,
        )

    return prompt_fn


validate_fusion_aya_prompt_parts()


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"fusion_aya_math:{language.value}:gen",
        prompt_function=fusion_aya_cot_prompt(language),
        hf_repo="tiny-aya-math-edition/fusion-aya-math-bench",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("benchmark",),
        few_shots_split=None,
        generation_size=4096,
        metrics=[
            Metrics.expr_gold_metric,
        ],
        stop_sequence=fusion_aya_stop_sequences(language),
    )
    for language in _LANGUAGES
]
