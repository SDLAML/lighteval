"""
name:
FinEval

dataset:
SUFE-AIFLM-Lab/FinEval

abstract:
FinEval is a Chinese financial knowledge benchmark covering finance, economics,
accounting, and professional certificates, drawn from university-level exams.
Single-language Chinese task — no language suffix in task name.

languages:
chinese

tags:
finance, knowledge, multilingual, multiple-choice, qa

paper:
https://arxiv.org/abs/2308.09975
"""

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


def _fineval_adapter(line):
    choices = [line["A"], line["B"], line["C"], line["D"]]
    gold = list("ABCD").index(line["answer"])
    return {"question": line["question"], "choices": choices, "gold_idx": gold}


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"fineval:{suffix}",
        prompt_function=get_mcq_prompt_function(Language.CHINESE, _fineval_adapter, formulation=formulation),
        hf_repo="SUFE-AIFLM-Lab/FinEval",
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=gen,
        metrics=metrics,
        stop_sequence=["\n"],
        version=0,
    )
    for suffix, formulation, metrics, gen in [
        ("cf",     CFFormulation(),  _CF_METRICS,             -1),
        ("mcf",    MCFFormulation(), _MCF_METRICS,            -1),
        ("mcf_em", MCFFormulation(), [Metrics.exact_match],    1),
    ]
]
