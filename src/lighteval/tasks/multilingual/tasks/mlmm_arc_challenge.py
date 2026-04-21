"""
name:
Mlmm Arc Challenge

dataset:
alexandrainst/m_arc

abstract:
ARC (AI2 Reasoning Challenge) is a dataset for question answering that requires
reasoning. It consists of multiple-choice science questions from 3rd to 9th
grade exams. The dataset is split into two parts: ARC-Easy and ARC-Challenge.
ARC-Easy contains questions that can be answered correctly by both humans and
simple baseline models. ARC-Challenge contains questions that are difficult for
both humans and current AI systems. Similar to MMLU, ARC tasks uses PMI
normalization by default but only for the challenge set. This multilingual
version is a machine-translated version of ARC maintained by the Alexandra
Institute.

languages:
arabic, armenian, basque, bengali, catalan, chinese, croatian, danish, dutch,
english, french, german, gujarati, hindi, hungarian, icelandic, indonesian,
italian, kannada, malayalam, marathi, nepali, norwegian, portuguese, romanian,
russian, serbian, slovak, spanish, swedish, tamil, telugu, ukrainian,
vietnamese

tags:
multilingual, multiple-choice, reasoning

paper:
https://huggingface.co/datasets/alexandrainst/m_arc
"""

from string import ascii_uppercase

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


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
        name=f"mlmm_arc_{language.value}_{formulation.name.lower()}:challenge",
        prompt_function=get_mcq_prompt_function(
            language,
            _m_arc_adapter,
            formulation=formulation,
        ),
        hf_repo="alexandrainst/m_arc",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                # LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for language in [
        Language.ARABIC,
        Language.ARMENIAN,
        Language.BASQUE,
        Language.BENGALI,
        Language.CATALAN,
        Language.CHINESE,
        Language.CROATIAN,
        Language.DANISH,
        Language.DUTCH,
        Language.ENGLISH,
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
