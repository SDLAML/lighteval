"""
name:
Global Mmlu Lite

dataset:
CohereLabs/Global-MMLU-Lite

abstract:
A lighter, culturally-annotated subset of MMLU covering 18 languages total;
17 non-English languages evaluated here (400 test samples and 215 dev samples
per language across 43 subjects). Designed for quick multilingual MMLU-style
evaluation.

English is excluded — use mmlu.py for English evaluation.

Metrics:
- :cf  — completion formulation; reports acc, acc_norm (char), target_bpb
- :mcf — multiple-choice formulation; reports acc, acc_norm (char)

languages:
arabic, bengali, welsh, german, spanish, french, hindi, indonesian, italian,
japanese, korean, burmese, portuguese, albanian, swahili, yoruba, chinese

tags:
knowledge, multilingual, multiple-choice

paper:
https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite
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

# 18 languages in Global-MMLU-Lite, English excluded
_LANGUAGES = [
    Language.ARABIC,        # ar
    Language.BENGALI,       # bn
    Language.WELSH,         # cy
    Language.GERMAN,        # de
    Language.SPANISH,       # es
    Language.FRENCH,        # fr
    Language.HINDI,         # hi
    Language.INDONESIAN,    # id
    Language.ITALIAN,       # it
    Language.JAPANESE,      # ja
    Language.KOREAN,        # ko
    Language.BURMESE,       # my
    Language.PORTUGUESE,    # pt
    Language.ALBANIAN,      # sq
    Language.SWAHILI,       # sw
    Language.YORUBA,        # yo
    Language.CHINESE,       # zh
]


def _global_mmlu_lite_adapter(line):
    return {
        "question": line["question"],
        "choices": [line["option_a"], line["option_b"], line["option_c"], line["option_d"]],
        "gold_idx": ascii_uppercase.index(line["answer"]),
    }


def _valid_options_filter(x):
    """Skip rows where any option is missing."""
    return all(x[f"option_{opt}"] is not None and x[f"option_{opt}"].strip() for opt in "abcd")


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"global_mmlu_lite:{language.value}:{suffix}",
        prompt_function=get_mcq_prompt_function(
            language,
            _global_mmlu_lite_adapter,
            formulation=formulation,
        ),
        hf_repo="CohereLabs/Global-MMLU-Lite",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=_valid_options_filter,
        metrics=metrics,
    )
    for language in _LANGUAGES
    for suffix, formulation, metrics in [
        ("cf", CFFormulation(), _CF_METRICS),
        ("mcf", MCFFormulation(), _MCF_METRICS),
    ]
]

# Greedy variant: MCF-style prompt, generate up to 5 tokens, exact match
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"global_mmlu_lite:{language.value}:mcf_em",
        prompt_function=get_mcq_prompt_function(
            language,
            _global_mmlu_lite_adapter,
            formulation=MCFFormulation(),
        ),
        hf_repo="CohereLabs/Global-MMLU-Lite",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=_valid_options_filter,
        generation_size=5,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for language in _LANGUAGES
]
