"""
name:
MedExpQA

dataset:
HiTZ/MedExpQA

abstract:
MedExpQA is a multilingual medical expert QA benchmark based on Spanish
board-exam style questions, with translations to French, Italian, and English.
Each question has 4-5 options with a single correct answer.

languages:
english, french, italian, spanish

tags:
medical, multilingual, multiple-choice, qa

paper:
https://arxiv.org/abs/2307.00099
"""

import ast

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

# (lighteval Language, HF config name)
_LANGUAGES = [
    (Language.SPANISH, "es"),
    (Language.FRENCH,  "fr"),
    (Language.ITALIAN, "it"),
    (Language.ENGLISH, "en"),
]


def _make_adapter():
    def adapter(line):
        opts = ast.literal_eval(line["options"]) if isinstance(line["options"], str) else line["options"]
        keys = sorted(opts.keys(), key=lambda x: int(x))
        choices = [opts[k] for k in keys]
        gold = keys.index(str(int(line["correct_option"])))
        return {"question": line["full_question"], "choices": choices, "gold_idx": gold}
    return adapter


_adapter = _make_adapter()

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"medexpqa:{language.value}:{suffix}",
        prompt_function=get_mcq_prompt_function(language, _adapter, formulation=formulation),
        hf_repo="HiTZ/MedExpQA",
        hf_subset=hf_config,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=gen,
        metrics=metrics,
        stop_sequence=["\n"],
        version=0,
    )
    for language, hf_config in _LANGUAGES
    for suffix, formulation, metrics, gen in [
        ("cf",     CFFormulation(),  _CF_METRICS,             -1),
        ("mcf",    MCFFormulation(), _MCF_METRICS,            -1),
        ("mcf_em", MCFFormulation(), [Metrics.exact_match],    1),
    ]
]
