"""
name:
Swag

dataset:
allenai/swag

abstract:
The dataset consists of 113k multiple choice questions about grounded situations
(73k training, 20k validation, 20k test). Each question is a video caption from
LSMDC or ActivityNet Captions, with four answer choices about what might happen
next in the scene. The correct answer is the (real) video caption for the next
event in the video; the three incorrect answers are adversarially generated and
human verified, so as to fool machines but not humans. SWAG aims to be a
benchmark for evaluating grounded commonsense NLI and for learning
representations.

languages:
english

tags:
narrative, reasoning

paper:
https://arxiv.org/abs/1808.05326
"""

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

_ENDINGS = ["ending0", "ending1", "ending2", "ending3"]


def swag_cf_prompt(line, task_name: str = None):
    """CF variant: score full completion texts via logprobs."""
    choices = [line[e] for e in _ENDINGS]
    return Doc(
        task_name=task_name,
        query=line["startphrase"],
        choices=[" " + c for c in choices],
        gold_index=int(line["label"]),
    )


def swag_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C/D options, score label tokens via logprobs."""
    choices = [line[e] for e in _ENDINGS]
    options = "\n".join(f" {l}. {c}" for l, c in zip("ABCD", choices))
    query = f"{line['startphrase']}\n{options}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=int(line["label"]),
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="swag:cf",
        prompt_function=swag_cf_prompt,
        hf_repo="allenai/swag",
        hf_subset="regular",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=_CF_METRICS,
        stop_sequence=["\n"],
        version=1,
    ),
    LightevalTaskConfig(
        name="swag:mcf",
        prompt_function=swag_mcf_prompt,
        hf_repo="allenai/swag",
        hf_subset="regular",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=1,
    ),
    LightevalTaskConfig(
        name="swag:mcf_em",
        prompt_function=swag_mcf_prompt,
        hf_repo="allenai/swag",
        hf_subset="regular",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=1,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=1,
    ),
]
