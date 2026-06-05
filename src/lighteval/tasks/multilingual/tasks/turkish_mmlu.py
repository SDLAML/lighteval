"""
name:
Turkish Mmlu

dataset:
AYueksel/TurkishMMLU

abstract:
TurkishMMLU is a Turkish-language multiple-choice benchmark modelled after
MMLU, covering 9 school subjects.

languages:
turkish

tags:
knowledge, multilingual, multiple-choice

paper:
"""

from string import ascii_uppercase

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

# HF subset names (capitalized); task names use lowercase.
TURKISH_MMLU_SUBSETS = [
    "Biology",
    "Chemistry",
    "Geography",
    "History",
    "Mathematics",
    "Philosophy",
    "Physics",
    "Religion_and_Ethics",
    "Turkish_Language_and_Literature",
]


def _adapter(line):
    return {
        "question": line["question"],
        "choices": line["choices"],
        "gold_idx": ascii_uppercase.index(line["answer"]),
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"turkishmmlu:{subset.lower()}:{suffix}",
        prompt_function=get_mcq_prompt_function(Language.TURKISH, _adapter, formulation=formulation),
        hf_repo="AYueksel/TurkishMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=gen_size,
        metrics=metrics,
        stop_sequence=["\n"],
        version=1,
    )
    for subset in TURKISH_MMLU_SUBSETS
    for suffix, formulation, metrics, gen_size in [
        ("cf",     CFFormulation(),  _CF_METRICS,             -1),
        ("mcf",    MCFFormulation(), _MCF_METRICS,            -1),
        ("mcf_em", MCFFormulation(), [Metrics.exact_match],    1),
    ]
]
