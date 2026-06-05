"""
name:
Exams

dataset:
mhardalov/exams

abstract:
EXAMS is a multilingual benchmark for school-level subject knowledge across
16 languages and multiple subjects per language.

languages:
albanian, arabic, bulgarian, croatian, french, german, hungarian, italian,
lithuanian, macedonian, polish, portuguese, serbian, spanish, turkish,
vietnamese

tags:
knowledge, multilingual, multiple-choice

paper:
https://arxiv.org/abs/2011.03080
"""

from langcodes import Language as LangCodeLanguage
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

# Languages with EXAMS data; all subjects for each language are aggregated.
_LANGUAGES = [
    Language.ARABIC,
    Language.BULGARIAN,
    Language.CROATIAN,
    Language.HUNGARIAN,
    Language.ITALIAN,
    Language.SERBIAN,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
    Language.LITHUANIAN,
    Language.ALBANIAN,
    Language.MACEDONIAN,
    Language.TURKISH,
    Language.POLISH,
    Language.PORTUGUESE,
    Language.VIETNAMESE,
]


def _lang_name(language: Language) -> str:
    return LangCodeLanguage(standardize_tag(language.value)).language_name()


def _make_filter(language: Language):
    lang_name = _lang_name(language)
    return lambda line: (
        line["answerKey"] != "@"
        and line["info"]["language"] == lang_name
    )


def _adapter(line):
    return {
        "question": line["question"]["stem"],
        "choices": line["question"]["choices"]["text"],
        "gold_idx": line["question"]["choices"]["label"].index(line["answerKey"]),
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"exams:{language.value}:{suffix}",
        prompt_function=get_mcq_prompt_function(language, _adapter, formulation=formulation),
        hf_repo="mhardalov/exams",
        hf_subset="multilingual",
        hf_filter=_make_filter(language),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=gen_size,
        metrics=metrics,
        stop_sequence=["\n"],
        version=1,
    )
    for language in _LANGUAGES
    for suffix, formulation, metrics, gen_size in [
        ("cf",     CFFormulation(),  _CF_METRICS,             -1),
        ("mcf",    MCFFormulation(), _MCF_METRICS,            -1),
        ("mcf_em", MCFFormulation(), [Metrics.exact_match],    1),
    ]
]
