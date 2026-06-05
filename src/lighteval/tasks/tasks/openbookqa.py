"""
name:
Openbookqa

dataset:
allenai/openbookqa

abstract:
OpenBookQA is a question-answering dataset modeled after open-book exams for
assessing human understanding of a subject. It contains multiple-choice
questions that require combining facts from a given open book with broad common
knowledge. The task tests language models' ability to leverage provided
information and apply common sense reasoning.

languages:
english

tags:
multiple-choice, qa

paper:
https://arxiv.org/abs/1809.02789
"""

from string import ascii_uppercase

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


def _gold_index(line) -> int:
    return list(ascii_uppercase).index(line["answerKey"].strip())


def openbookqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C/D options in prompt, score label tokens via logprobs."""
    texts = line["choices"]["text"]
    labels = list(ascii_uppercase)[: len(texts)]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, texts))
    query = f"Question: {line['question_stem']}\n{options}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {l}" for l in labels],
        gold_index=_gold_index(line),
    )


def openbookqa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt with full answer texts as choices."""
    texts = line["choices"]["text"]
    query = f"Question: {line['question_stem']}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + c for c in texts],
        gold_index=_gold_index(line),
    )


# Greedy variant: MCF-style prompt, generate 1 token, exact match
openbookqa_mcf_em = LightevalTaskConfig(
    name="openbookqa:mcf_em",
    prompt_function=openbookqa_mcf_prompt,
    hf_repo="allenai/openbookqa",
    hf_subset="main",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
openbookqa_mcf = LightevalTaskConfig(
    name="openbookqa:mcf",
    prompt_function=openbookqa_mcf_prompt,
    hf_repo="allenai/openbookqa",
    hf_subset="main",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# CF variant: completion-style, logprob on full answer text + BPB on gold choice
openbookqa_cf = LightevalTaskConfig(
    name="openbookqa:cf",
    prompt_function=openbookqa_cf_prompt,
    hf_repo="allenai/openbookqa",
    hf_subset="main",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    openbookqa_mcf_em,
    openbookqa_mcf,
    openbookqa_cf,
]
