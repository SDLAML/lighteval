"""
name:
MMLU-ProX (English)

dataset:
li-lab/MMLU-ProX

abstract:
MMLU-ProX is a multilingual extension of MMLU-Pro. This file covers the
English ("en") subset only. It uses up to 10 answer options per question
(labels A–J), consistent with MMLU-Pro's harder format.

For multilingual evaluation see:
lighteval/tasks/multilingual/tasks/mmlu_prox.py

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://huggingface.co/datasets/li-lab/MMLU-ProX
"""

from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
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
    query += "".join([f"{lbl}. {opt}\n" for lbl, opt in zip(labels, options)])
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


mmlu_prox_cf = LightevalTaskConfig(
    name="mmlu_prox_eng:cf",
    prompt_function=mmlu_prox_cf_prompt,
    hf_repo="li-lab/MMLU-ProX",
    hf_subset="en",
    evaluation_splits=("test",),
    few_shots_split="validation",
    hf_filter=_valid_filter,
    metrics=_CF_METRICS,
    version=0,
)

mmlu_prox_mcf = LightevalTaskConfig(
    name="mmlu_prox_eng:mcf",
    prompt_function=mmlu_prox_mcf_prompt,
    hf_repo="li-lab/MMLU-ProX",
    hf_subset="en",
    evaluation_splits=("test",),
    few_shots_split="validation",
    hf_filter=_valid_filter,
    metrics=_MCF_METRICS,
    version=0,
)

# Greedy variant: MCF-style prompt, generate up to 5 tokens, exact match
mmlu_prox_mcf_em = LightevalTaskConfig(
    name="mmlu_prox_eng:mcf_em",
    prompt_function=mmlu_prox_mcf_prompt,
    hf_repo="li-lab/MMLU-ProX",
    hf_subset="en",
    evaluation_splits=("test",),
    few_shots_split="validation",
    hf_filter=_valid_filter,
    generation_size=5,
    stop_sequence=["\n"],
    metrics=[Metrics.exact_match],
    version=0,
)

TASKS_TABLE = [mmlu_prox_cf, mmlu_prox_mcf, mmlu_prox_mcf_em]
