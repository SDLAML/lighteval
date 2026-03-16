"""
name:
Lambada

dataset:
cimec/lambada

abstract:
LAMBADA is a benchmark for testing language models' ability to understand broad
narrative context. Each passage requires predicting its final word—easy for
humans given the full passage but impossible from just the last sentence.
Success demands long-range discourse comprehension.

languages:
english

tags:
language-modeling

paper:
https://arxiv.org/abs/1606.06031
"""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def lambada_prompt(line, task_name: str = None):
    """Standard LAMBADA prompt: context as query, last word as gold continuation."""
    query, choice = line["text"].rsplit(" ", 1)
    return Doc(
        task_name=task_name,
        query=query,
        gold_index=0,
        choices=[f" {choice}"],
    )


def lambada_cloze_prompt(line, task_name: str = None):
    """Cloze-style LAMBADA prompt with fill-in-the-blank indicator."""
    query, choice = line["text"].rsplit(" ", 1)
    return Doc(
        task_name=task_name,
        query=f"{query} ____. ->",
        gold_index=0,
        choices=[f" {choice}"],
    )


# CF variant: RC per-char (rank choice with per-character normalization) + BPB on gold
lambada_cf = LightevalTaskConfig(
    name="lambada:cf",
    prompt_function=lambada_prompt,
    hf_repo="cimec/lambada",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        Metrics.target_bits_per_byte,
    ],
    stop_sequence=["\n"],
    version=0,
)

# Cloze perplexity variant — logprob mode, cloze prompt (matches lm-eval lambada_standard_cloze_yaml)
lambada_standard_cloze = LightevalTaskConfig(
    name="lambada:standard_cloze",
    prompt_function=lambada_cloze_prompt,
    hf_repo="cimec/lambada",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.target_perplexity],
    stop_sequence=["\n"],
    version=0,
)

# OpenAI LAMBADA cloze variant — same cloze prompt but uses EleutherAI/lambada_openai dataset
# (matches lm-eval lambada_openai_cloze_yaml)
lambada_openai_cloze = LightevalTaskConfig(
    name="lambada:openai_cloze",
    prompt_function=lambada_cloze_prompt,
    hf_repo="EleutherAI/lambada_openai",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.target_perplexity],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    lambada_cf,
    lambada_standard_cloze,
    lambada_openai_cloze,
]
