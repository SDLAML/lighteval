"""
name:
Mlmm Hellaswag

dataset:
alexandrainst/m_hellaswag (most languages)
jon-tow/okapi_hellaswag (Chinese)

abstract:
Hellaswag is a commonsense reasoning task that requires models to complete a
given scenario with the most plausible ending. It tests the model's ability to
understand and reason about everyday situations and human behavior.
MLMM-Hellaswag: Multilingual adaptation of Hellaswag.

Refactored to use unified :cf / :mcf / :mcf_em suffixes consistent with English hellaswag.py.
HellaSwag has 4 candidate continuations; the MCF formulation labels them A–D and scores (or
generates) the label token. English is excluded — use hellaswag.py for English evaluation.

languages:
arabic, armenian, basque, bengali, catalan, chinese, croatian, danish, dutch,
french, german, gujarati, hindi, hungarian, icelandic, indonesian, italian,
kannada, malayalam, marathi, nepali, portuguese, romanian, russian,
serbian, slovak, spanish, swedish, tamil, telugu, ukrainian, vietnamese

tags:
multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2306.07610
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
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

# Languages covered by alexandrainst/m_hellaswag and jon-tow/okapi_hellaswag.
# English excluded — use hellaswag.py for English evaluation.
_LANGUAGES = [
    Language.ARABIC,
    Language.BENGALI,
    Language.CATALAN,
    Language.DANISH,
    Language.GERMAN,
    Language.SPANISH,
    Language.BASQUE,
    Language.FRENCH,
    Language.GUJARATI,
    Language.HINDI,
    Language.CROATIAN,
    Language.HUNGARIAN,
    Language.ARMENIAN,
    Language.INDONESIAN,
    Language.ICELANDIC,
    Language.ITALIAN,
    Language.KANNADA,
    Language.MALAYALAM,
    Language.MARATHI,
    Language.NEPALI,
    Language.DUTCH,
    Language.PORTUGUESE,
    Language.ROMANIAN,
    Language.RUSSIAN,
    Language.SLOVAK,
    Language.SERBIAN,
    Language.SWEDISH,
    Language.TAMIL,
    Language.TELUGU,
    Language.UKRAINIAN,
    Language.VIETNAMESE,
    Language.CHINESE,
]


def _hellaswag_adapter(line):
    return {
        "ctx_a": line["ctx_a"],
        "ctx_b": line["ctx_b"],
        "continuations": line["endings"],
        "gold_idx": int(line["label"]),
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag:{lang.value}:cf",
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=_hellaswag_adapter,
            formulation=CFFormulation(),
        ),
        hf_repo="alexandrainst/m_hellaswag" if lang != Language.CHINESE else "jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["val" if lang != Language.CHINESE else "validation"],
        hf_avail_splits=["val" if lang != Language.CHINESE else "validation"],
        few_shots_split=None,
        metrics=_CF_METRICS,
    )
    for lang in _LANGUAGES
]

# MCF variant: labeled options in prompt, score label tokens via logprobs
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag:{lang.value}:mcf",
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=_hellaswag_adapter,
            formulation=MCFFormulation(),
        ),
        hf_repo="alexandrainst/m_hellaswag" if lang != Language.CHINESE else "jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["val" if lang != Language.CHINESE else "validation"],
        hf_avail_splits=["val" if lang != Language.CHINESE else "validation"],
        few_shots_split=None,
        metrics=_MCF_METRICS,
    )
    for lang in _LANGUAGES
]

# Greedy variant: MCF-style prompt, generate 1 token, exact match
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag:{lang.value}:mcf_em",
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=_hellaswag_adapter,
            formulation=MCFFormulation(),
        ),
        hf_repo="alexandrainst/m_hellaswag" if lang != Language.CHINESE else "jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["val" if lang != Language.CHINESE else "validation"],
        hf_avail_splits=["val" if lang != Language.CHINESE else "validation"],
        few_shots_split=None,
        generation_size=1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for lang in _LANGUAGES
]
