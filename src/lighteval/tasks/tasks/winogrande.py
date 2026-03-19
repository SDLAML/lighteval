"""
name:
Winogrande

dataset:
allenai/winogrande

abstract:
WinoGrande is a new collection of 44k problems, inspired by Winograd Schema
Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the
scale and robustness against the dataset-specific bias. Formulated as a
fill-in-a-blank task with binary options, the goal is to choose the right option
for a given sentence which requires commonsense reasoning.

languages:
english

tags:
commonsense, multiple-choice

paper:
https://arxiv.org/abs/1907.10641
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
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


def winogrande_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style, score full answer texts via logprobs."""
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,
    )


def winogrande_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B options, score label tokens via logprobs."""
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    query = query.strip()
    opt1 = f"{line['option1']} {end_of_target}"
    opt2 = f"{line['option2']} {end_of_target}"
    query = f"{query}\nA. {opt1}\nB. {opt2}\nAnswer:"
    gold_ix = int(line["answer"]) - 1 if line["answer"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B"],
        gold_index=gold_ix,
    )


# CF variant: completion-style, logprob on full answer text + BPB on gold choice
winogrande_cf = LightevalTaskConfig(
    name="winogrande:cf",
    prompt_function=winogrande_cf_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
winogrande_mcf = LightevalTaskConfig(
    name="winogrande:mcf",
    prompt_function=winogrande_mcf_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# Greedy variant: MCF-style prompt, generate 1 token, exact match
winogrande_mcf_em = LightevalTaskConfig(
    name="winogrande:mcf_em",
    prompt_function=winogrande_mcf_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    winogrande_cf,
    winogrande_mcf,
    winogrande_mcf_em,
]
