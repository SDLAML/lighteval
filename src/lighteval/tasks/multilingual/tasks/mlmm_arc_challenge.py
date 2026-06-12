"""
name:
Mlmm Arc Challenge

dataset:
alexandrainst/m_arc

abstract:
ARC (AI2 Reasoning Challenge) is a dataset for question answering that requires
reasoning. It consists of multiple-choice science questions from 3rd to 9th
grade exams. The dataset is split into two parts: ARC-Easy and ARC-Challenge.
ARC-Challenge contains questions that are difficult for both humans and current
AI systems.

Refactored to use unified :cf / :mcf suffixes consistent with English arc.py.
English is excluded — use arc.py for English evaluation.

languages:
arabic, bengali, catalan, chinese, croatian, danish, dutch, french, german,
hindi, hungarian, indonesian, italian, kannada, malayalam, marathi, nepali,
romanian, russian, serbian, slovak, spanish, tamil, telugu, ukrainian,
vietnamese

tags:
multilingual, multiple-choice, reasoning

paper:
https://github.com/nlp-uoregon/mlmm-evaluation
"""

from string import ascii_uppercase

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
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
    Language.ARMENIAN,
    Language.BASQUE,
    Language.BENGALI,
    Language.CATALAN,
    Language.CHINESE,
    Language.CROATIAN,
    Language.DANISH,
    Language.DUTCH,
    # Language.ENGLISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.GUJARATI,
    Language.HINDI,
    Language.HUNGARIAN,
    Language.ICELANDIC,
    Language.INDONESIAN,
    Language.ITALIAN,
    Language.KANNADA,
    Language.MALAYALAM,
    Language.MARATHI,
    Language.NEPALI,
    Language.NORWEGIAN,
    Language.PORTUGUESE,
    Language.ROMANIAN,
    Language.RUSSIAN,
    Language.SERBIAN,
    Language.SLOVAK,
    Language.SPANISH,
    Language.SWEDISH,
    Language.TAMIL,
    Language.TELUGU,
    Language.UKRAINIAN,
    Language.VIETNAMESE,
]

def _m_arc_adapter(line):
    raw_choices = [line.get(f"option_{letter}") for letter in "abcde"]
    choices = [c for c in raw_choices if c is not None]
    return {
        "question": line["instruction"],
        "choices": choices,
        "gold_idx": ascii_uppercase.index(line["answer"].strip().upper()),
    }

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mlmm_arc_challenge:{language.value}:{suffix}",
        prompt_function=get_mcq_prompt_function(
            language,
            _m_arc_adapter,
            formulation=formulation,
        ),
        hf_repo="alexandrainst/m_arc",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        metrics=metrics,
    )
    for language in _LANGUAGES
    for suffix, formulation, metrics in [
        ("cf", CFFormulation(), _CF_METRICS),
        ("mcf", MCFFormulation(), _MCF_METRICS),
    ]
]

# Greedy variant: MCF-style prompt, generate 1 token, exact match
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mlmm_arc_challenge:{language.value}:mcf_em",
        prompt_function=get_mcq_prompt_function(
            language,
            _m_arc_adapter,
            formulation=MCFFormulation(),
        ),
        hf_repo="alexandrainst/m_arc",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for language in _LANGUAGES
]
