"""
name:
AgriEval + CROP

datasets:
PaperHarvester/AgriEval
AI4Agr/CROP-benchmark

abstract:
Chinese agricultural knowledge benchmarks. AgriEval covers plant science,
animal production, and related domains (variable 2–7 choices). CROP covers
crop science with 4-choice questions. Both are Chinese-language.

Single-language Chinese tasks — no language suffix in task name.

languages:
chinese

tags:
agriculture, knowledge, multilingual, multiple-choice, qa

paper:
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


def _agrieval_adapter(line):
    opts = line["options"]
    keys = sorted(opts.keys())
    choices = [opts[k] for k in keys]
    gold = keys.index(line["answer"])
    return {"question": line["question"], "choices": choices, "gold_idx": gold}


def _crop_adapter(line):
    choices = [line[f"Option {l}"] for l in "ABCD"]
    gold = list("ABCD").index(line["Answer"])
    return {"question": line["Question"], "choices": choices, "gold_idx": gold}


def _configs(name, repo, subset, split, adapter, fs_select):
    return [
        LightevalTaskConfig(
            name=f"{name}:{suffix}",
            prompt_function=get_mcq_prompt_function(Language.CHINESE, adapter, formulation=formulation),
            hf_repo=repo,
            hf_subset=subset,
            hf_avail_splits=[split],
            evaluation_splits=[split],
            few_shots_split=split,
            few_shots_select=fs_select,
            # single-choice only; AgriEval has 单选/多选
            hf_filter=(lambda line: line.get("question_type") == "单选") if name == "agrieval" else None,
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


TASKS_TABLE = (
    _configs("agrieval", "PaperHarvester/AgriEval", "default", "train", _agrieval_adapter, "random_sampling")
    + _configs("crop",     "AI4Agr/CROP-benchmark",  "default", "test",  _crop_adapter,    "random_sampling")
)
