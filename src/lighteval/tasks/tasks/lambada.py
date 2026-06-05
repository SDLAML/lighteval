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

import hashlib
import random

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

# Distractor pool for lambada:cf — loaded lazily on first prompt call.
# Contains all last words from the test split; used to sample 3 distractors
# per example via a deterministic hash of the passage text.
_LAMBADA_LAST_WORDS = None


def _get_lambada_last_words():
    global _LAMBADA_LAST_WORDS
    if _LAMBADA_LAST_WORDS is None:
        import datasets as _hf_datasets
        ds = _hf_datasets.load_dataset("cimec/lambada", "plain_text", split="test")
        _LAMBADA_LAST_WORDS = [text.rsplit(" ", 1)[1] for text in ds["text"]]
    return _LAMBADA_LAST_WORDS


def lambada_prompt(line, task_name: str = None):
    """Standard LAMBADA prompt: context as query, last word as gold continuation."""
    query, choice = line["text"].rsplit(" ", 1)
    return Doc(
        task_name=task_name,
        query=query,
        gold_index=0,
        choices=[f" {choice}"],
    )


def lambada_multichoice_prompt(line, task_name: str = None):
    """RC_per-char prompt: gold last word vs 3 distractors sampled from the test set.

    Distractors are chosen deterministically by hashing the passage text, so
    results are reproducible without storing a precomputed mapping file.
    """
    query, gold = line["text"].rsplit(" ", 1)
    last_words = _get_lambada_last_words()
    seed = int(hashlib.md5(line["text"].encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    pool = [w for w in last_words if w != gold]
    distractors = rng.sample(pool, 3)
    all_choices = distractors + [gold]
    rng.shuffle(all_choices)
    gold_ix = all_choices.index(gold)
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + c for c in all_choices],
        gold_index=gold_ix,
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


# BPB variant: single-choice, score gold continuation only (no acc ranking)
lambada_bpb = LightevalTaskConfig(
    name="lambada:bpb",
    prompt_function=lambada_prompt,
    hf_repo="cimec/lambada",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.target_bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

# CF variant: multi-choice RC per-char acc (gold last word vs 3 sampled distractors) + BPB
lambada_cf = LightevalTaskConfig(
    name="lambada:cf",
    prompt_function=lambada_multichoice_prompt,
    hf_repo="cimec/lambada",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        LogLikelihoodAccMetric(normalization=LogProbCharNorm())
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
    lambada_bpb,
    lambada_cf,
    lambada_standard_cloze,
    lambada_openai_cloze,
]
