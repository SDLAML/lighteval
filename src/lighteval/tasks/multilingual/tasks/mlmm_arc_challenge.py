"""
name:
Mlmm Arc Challenge

dataset:
jon-tow/okapi_arc_challenge

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

# Languages covered by jon-tow/okapi_arc_challenge (English excluded)
_LANGUAGES = [
    Language.RUSSIAN,
    Language.GERMAN,
    Language.CHINESE,
    Language.FRENCH,
    Language.SPANISH,
    Language.ITALIAN,
    Language.DUTCH,
    Language.VIETNAMESE,
    Language.INDONESIAN,
    Language.ARABIC,
    Language.HUNGARIAN,
    Language.ROMANIAN,
    Language.DANISH,
    Language.SLOVAK,
    Language.UKRAINIAN,
    Language.CATALAN,
    Language.SERBIAN,
    Language.CROATIAN,
    Language.HINDI,
    Language.BENGALI,
    Language.TAMIL,
    Language.NEPALI,
    Language.MALAYALAM,
    Language.MARATHI,
    Language.TELUGU,
    Language.KANNADA,
]


def _arc_adapter(line):
    if "question" in line and "choices" in line:
        choices = line["choices"]["text"]
        answer_key = line["answerKey"]
    else:
        choices = [
            line[key]
            for key in ("option_a", "option_b", "option_c", "option_d", "option_e")
            if line.get(key)
        ]
        answer_key = line["answer"]
        return {
            "question": line["instruction"],
            "choices": choices,
            "gold_idx": int(answer_key) - 1
            if answer_key.isdigit()
            else ascii_uppercase.index(answer_key),
        }

    return {
        "question": line["question"],
        "choices": choices,
        "gold_idx": int(answer_key) - 1
        if answer_key.isdigit()
        else ascii_uppercase.index(answer_key),
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mlmm_arc:{language.value}:{suffix}",
        prompt_function=get_mcq_prompt_function(
            language,
            _arc_adapter,
            formulation=formulation,
        ),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(language.value),
        hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
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
        name=f"mlmm_arc:{language.value}:mcf_em",
        prompt_function=get_mcq_prompt_function(
            language,
            _arc_adapter,
            formulation=MCFFormulation(),
        ),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(language.value),
        hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for language in _LANGUAGES
]
