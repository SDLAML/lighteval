"""
name:
MMLU-ProX (Multilingual MMLU-Pro)

dataset:
li-lab/MMLU-ProX

abstract:
MMLU-ProX is a multilingual extension of MMLU-Pro covering 29 languages. It
uses up to 10 answer options per question (labels A–J), making it more
challenging than standard 4-option MMLU.

English is excluded — use mmlu_pro.py for English evaluation.

Metrics:
- :cf  — completion formulation; scores each non-null option text; reports
          acc, acc_norm (char), target_bpb
- :mcf — multiple-choice formulation; scores label tokens A–J; reports
          acc, acc_norm (char)

languages:
afrikaans, arabic, bengali, czech, german, spanish, french, hindi, hungarian,
indonesian, italian, japanese, korean, marathi, nepali, portuguese, russian,
serbian, swahili, telugu, thai, ukrainian, urdu, vietnamese, wolof, yoruba,
chinese, zulu

tags:
knowledge, multilingual, multiple-choice

paper:
https://huggingface.co/datasets/li-lab/MMLU-ProX
"""

from string import ascii_uppercase

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
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

# MMLU-ProX languages (English excluded — use mmlu_pro.py for English)
_LANGUAGES = [
    Language.AFRIKAANS,   # af
    Language.ARABIC,      # ar
    Language.BENGALI,     # bn
    Language.CZECH,       # cs
    Language.GERMAN,      # de
    Language.SPANISH,     # es
    Language.FRENCH,      # fr
    Language.HINDI,       # hi
    Language.HUNGARIAN,   # hu
    Language.INDONESIAN,  # id
    Language.ITALIAN,     # it
    Language.JAPANESE,    # ja
    Language.KOREAN,      # ko
    Language.MARATHI,     # mr
    Language.NEPALI,      # ne
    Language.PORTUGUESE,  # pt
    Language.RUSSIAN,     # ru
    Language.SERBIAN,     # sr
    Language.SWAHILI,     # sw
    Language.TELUGU,      # te
    Language.THAI,        # th
    Language.UKRAINIAN,   # uk
    Language.URDU,        # ur
    Language.VIETNAMESE,  # vi
    Language.WOLOF,       # wo
    Language.YORUBA,      # yo
    Language.CHINESE,     # zh
    Language.ZULU,        # zu
]

# option_0..option_9 column names
_OPTION_COLS = [f"option_{i}" for i in range(10)]


def _get_options(line):
    """Return list of non-null options from option_0..option_9."""
    return [line[col] for col in _OPTION_COLS if line.get(col) is not None and str(line[col]).strip()]


def mmlu_prox_cf_prompt(line, task_name: str = None):
    """CF: score each non-null option text directly."""
    options = _get_options(line)
    if not options:
        return None
    choices = [c if c and c[0].isspace() else " " + c for c in options]
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question'].strip()}\nAnswer:",
        choices=choices,
        gold_index=line["answer_index"],
    )


def mmlu_prox_mcf_prompt(line, task_name: str = None):
    """MCF: show labeled options A–J, score label tokens only."""
    options = _get_options(line)
    if not options:
        return None
    labels = list(ascii_uppercase[: len(options)])
    query = f"Question: {line['question'].strip()}\n"
    query += "".join([f" {lbl}. {opt}\n" for lbl, opt in zip(labels, options)])
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + lbl for lbl in labels],
        gold_index=line["answer_index"],
    )


def _valid_filter(x):
    """Skip rows where no options are present."""
    return any(
        x.get(col) is not None and str(x[col]).strip()
        for col in _OPTION_COLS
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu_prox:{language.value}:{suffix}",
        prompt_function=prompt_fn,
        hf_repo="li-lab/MMLU-ProX",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="validation",
        hf_filter=_valid_filter,
        metrics=metrics,
    )
    for language in _LANGUAGES
    for suffix, prompt_fn, metrics in [
        ("cf", mmlu_prox_cf_prompt, _CF_METRICS),
        ("mcf", mmlu_prox_mcf_prompt, _MCF_METRICS),
    ]
]

# Greedy variant: MCF-style prompt, generate up to 5 tokens, exact match
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_prox:{language.value}:mcf_em",
        prompt_function=mmlu_prox_mcf_prompt,
        hf_repo="li-lab/MMLU-ProX",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="validation",
        hf_filter=_valid_filter,
        generation_size=5,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for language in _LANGUAGES
]
