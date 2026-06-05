"""
name:
LexEval Legal Evaluation QA

dataset:
titaneval_local (local parquet — data/titaneval/lexeval.parquet)

abstract:
Legal evaluation multiple-choice benchmark (Chinese). 10,920 questions
from TitanEval-MCQ, 0-shot.

languages:
chinese

tags:
law, multiple-choice, qa, chinese
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


def _adapter(line):
    return {
        "question": line["question"],
        "choices": line["choices"],
        "gold_idx": line["answer_index"],
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"lexeval:{suffix}",
        prompt_function=get_mcq_prompt_function(Language.CHINESE, _adapter, formulation=formulation),
        hf_repo="titaneval_local",
        hf_subset="lexeval",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=gen_size,
        metrics=metrics,
        stop_sequence=["\n"],
        version=0,
    )
    for suffix, formulation, metrics, gen_size in [
        ("cf",     CFFormulation(),  _CF_METRICS,              -1),
        ("mcf",    MCFFormulation(), _MCF_METRICS,             -1),
        ("mcf_em", MCFFormulation(), [Metrics.exact_match],     1),
    ]
]
