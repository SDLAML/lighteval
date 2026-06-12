"""
name:
Xcopa

dataset:
xcopa

abstract:
COPA (Choice of Plausible Alternatives) tasks involve determining the most
plausible cause or effect for a given premise. These tasks test common sense
reasoning and causal inference abilities. XCOPA: Cross-lingual Choice of
Plausible Alternatives.

Refactored to use unified :cf / :mcf / :mcf_em suffixes consistent with the
multilingual naming convention. English is not part of XCOPA — use the English
COPA task for English evaluation.

languages:
arabic, chinese, estonian, haitian, indonesian, italian, quechua, swahili,
tamil, thai, turkish, vietnamese

tags:
multilingual, multiple-choice, narrative, reasoning

paper:
https://aclanthology.org/2020.emnlp-main.185/
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.copa import get_copa_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]

_LANGUAGES = [
    Language.ARABIC,
    Language.ESTONIAN,
    Language.INDONESIAN,
    Language.ITALIAN,
    Language.SWAHILI,
    Language.TAMIL,
    Language.THAI,
    Language.TURKISH,
    Language.VIETNAMESE,
    Language.CHINESE,
    Language.HAITIAN,
    Language.QUECHUA,
]


def _adapter(line):
    return {
        "context": line["premise"],
        "cause_effect": line["question"],
        "continuations": [line["choice1"], line["choice2"]],
        "gold_idx": int(line["label"]),
    }


def _hf_repo(language: Language) -> str:
    return "OALL/AlGhafa-Arabic-LLM-Benchmark-Translated" if language == Language.ARABIC else "cambridgeltl/xcopa"


# Dataset subset codes that differ from standardize_tag(language.value).
_SUBSET_OVERRIDES = {
    Language.ARABIC: "copa_ext_ar",  # different repo (OALL), different subset name
    Language.HAITIAN: "ht",          # standardize_tag("hti") -> "hti", dataset uses "ht"
}


def _hf_subset(language: Language) -> str:
    return _SUBSET_OVERRIDES.get(language, standardize_tag(language.value))


def _supports_mcf(language: Language) -> bool:
    """MCF / mcf_em build a labelled prompt that needs the localized 'answer' word.

    A few low-resource XCOPA languages (Haitian, Quechua) have no such translation
    literal, so MCF can't be formed for them — we keep CF (which needs no label) for
    all languages and only emit MCF/mcf_em where the literals exist.
    """
    lit = TRANSLATION_LITERALS[language]
    try:
        return bool(lit.answer) and bool(lit.question_word)
    except AttributeError:
        return False


_MCF_LANGUAGES = [language for language in _LANGUAGES if _supports_mcf(language)]


# CF for all languages (no localized labels needed).
TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"xcopa:{language.value}:cf",
        prompt_function=get_copa_prompt_function(language, adapter=_adapter, formulation=CFFormulation()),
        hf_repo=_hf_repo(language),
        hf_subset=_hf_subset(language),
        evaluation_splits=["test"],
        few_shots_split="validation",
        metrics=_CF_METRICS,
    )
    for language in _LANGUAGES
]

# MCF only for languages with the required translation literals.
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"xcopa:{language.value}:mcf",
        prompt_function=get_copa_prompt_function(language, adapter=_adapter, formulation=MCFFormulation()),
        hf_repo=_hf_repo(language),
        hf_subset=_hf_subset(language),
        evaluation_splits=["test"],
        few_shots_split="validation",
        metrics=_MCF_METRICS,
    )
    for language in _MCF_LANGUAGES
]

# Greedy variant: MCF-style prompt, generate 1 token, exact match (same language set as MCF).
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"xcopa:{language.value}:mcf_em",
        prompt_function=get_copa_prompt_function(language, adapter=_adapter, formulation=MCFFormulation()),
        hf_repo=_hf_repo(language),
        hf_subset=_hf_subset(language),
        evaluation_splits=["test"],
        few_shots_split="validation",
        generation_size=1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for language in _MCF_LANGUAGES
]
