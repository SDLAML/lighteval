"""
name:
Global Mmlu

dataset:
CohereForAI/Global-MMLU

abstract:
Translated MMLU using both professional and non-professional translators.
Contains tags for cultural sensitivity.

Refactored to use unified :cf / :mcf suffixes with 3-part names (base:lang:suffix)
so group selectors like `global_mmlu:cf|5` expand automatically.
English is excluded — use mmlu.py for English evaluation.

Per-category metrics (STEM / Humanities / Social / Other) are reported via
MMLUCategoryGroupingCF / MCF, routed via doc.specific["subject"].

languages:
amharic, arabic, bengali, chinese, czech, dutch, french, german, greek,
hausa, hebrew, hindi, igbo, indonesian, italian, japanese, kirghiz, korean,
lithuanian, malagasy, malay, nepali, nyanja, persian, polish, portuguese,
romanian, russian, serbian, shona, sinhala, somali, spanish, swahili,
swedish, tagalog, telugu, turkish, ukrainian, vietnamese, yoruba

tags:
knowledge, multilingual, multiple-choice

paper:
https://huggingface.co/papers/2412.03304
"""

from string import ascii_uppercase

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import MMLUCategoryGroupingCF, MMLUCategoryGroupingMCF
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.utils.language import Language


# All available non-English configs in Global-MMLU. The dataset exposes
# language configs as BCP-47-style short tags (for example, `fa`, `fil`, `sw`)
# so the task names stay on Language enum values while hf_subset uses
# standardize_tag(language.value).
_LANGUAGES = [
    Language.AMHARIC,
    Language.ARABIC,
    Language.BENGALI,
    Language.CHINESE,
    Language.CZECH,
    Language.DUTCH,
    Language.FRENCH,
    Language.GERMAN,
    Language.GREEK,
    Language.HAUSA,
    Language.HEBREW,
    Language.HINDI,
    Language.IGBO,
    Language.INDONESIAN,
    Language.ITALIAN,
    Language.JAPANESE,
    Language.KIRGHIZ,
    Language.KOREAN,
    Language.LITHUANIAN,
    Language.MALAGASY,
    Language.MALAY,
    Language.NEPALI,
    Language.NYANJA,
    Language.PERSIAN,
    Language.SPANISH,
    Language.POLISH,
    Language.PORTUGUESE,
    Language.ROMANIAN,
    Language.RUSSIAN,
    Language.SERBIAN,
    Language.SHONA,
    Language.SINHALA,
    Language.SOMALI,
    Language.SWEDISH,
    Language.SWAHILI,
    Language.TAGALOG,
    Language.TELUGU,
    Language.TURKISH,
    Language.UKRAINIAN,
    Language.VIETNAMESE,
    Language.YORUBA,
]


def _make_global_mmlu_cf_prompt(language: Language):
    """CF prompt: score each full answer text. Sets doc.specific with subject."""
    inner = get_mcq_prompt_function(language, _cf_adapter, formulation=CFFormulation())

    def prompt_fn(line, task_name=None):
        doc = inner(line, task_name)
        if doc is not None:
            doc.specific = {"subject": (line.get("subject") or "").lower()}
        return doc

    return prompt_fn


def _make_global_mmlu_mcf_prompt(language: Language):
    """MCF prompt: score label tokens A/B/C/D. Sets doc.specific with subject."""
    inner = get_mcq_prompt_function(language, _mcf_adapter, formulation=MCFFormulation())

    def prompt_fn(line, task_name=None):
        doc = inner(line, task_name)
        if doc is not None:
            doc.specific = {"subject": (line.get("subject") or "").lower()}
        return doc

    return prompt_fn


def _cf_adapter(line):
    choices = [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]
    if any(c is None or not str(c).strip() for c in choices):
        return None
    return {
        "question": line["question"],
        "choices": choices,
        "gold_idx": ascii_uppercase.index(line["answer"]),
    }


def _mcf_adapter(line):
    choices = [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]
    if any(c is None or not str(c).strip() for c in choices):
        return None
    return {
        "question": line["question"],
        "choices": choices,
        "gold_idx": ascii_uppercase.index(line["answer"]),
    }


_MMLU_CF_METRICS = [MMLUCategoryGroupingCF]
_MMLU_MCF_METRICS = [MMLUCategoryGroupingMCF]


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"global_mmlu:{language.value}:{suffix}",
        prompt_function=prompt_fn(language),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        metrics=metrics,
    )
    for language in _LANGUAGES
    for suffix, prompt_fn, metrics in [
        ("cf",  _make_global_mmlu_cf_prompt,  _MMLU_CF_METRICS),
        ("mcf", _make_global_mmlu_mcf_prompt, _MMLU_MCF_METRICS),
    ]
]

# Greedy variant: MCF-style prompt, generate up to 5 tokens, exact match
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"global_mmlu:{language.value}:mcf_em",
        prompt_function=_make_global_mmlu_mcf_prompt(language),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=5,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for language in _LANGUAGES
]
