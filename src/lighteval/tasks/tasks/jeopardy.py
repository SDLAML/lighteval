"""
name:
Jeopardy

dataset:
soldni/jeopardy (mosaicml_gauntlet subset, 2117 questions)
allenai/jeopardy_mc (multiple-choice reformulation)

abstract:
Jeopardy consists of 2,117 Jeopardy questions separated into 5 categories:
Literature, American History, World History, Word Origins, and Science.
Sourced from the MosaicML Gauntlet subset of soldni/jeopardy.

languages:
english

tags:
knowledge, qa, trivia

paper:
"""

import re

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def _parse_context(context: str):
    """Parse 'CATEGORY: question text' into (category, question)."""
    m = re.findall(r"(.*?):\s*(.*)", context)
    if m:
        return m[0]
    return ("", context)


# ── GenQA / BPB  (soldni/jeopardy, mosaicml_gauntlet subset) ──────────────────

def jeopardy_gen_prompt(line, task_name: str = None):
    gold_text = line["continuation"]
    if not gold_text:
        return None
    category, question = _parse_context(line["context"])
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=f"Category: {category}\nQuestion: {question}\nAnswer:",
        choices=[f"{' ' if is_few_shots else ''}{gold_text}"],
        gold_index=0,
    )


def jeopardy_bpb_prompt(line, task_name: str = None):
    gold_text = line["continuation"]
    if not gold_text:
        return None
    category, question = _parse_context(line["context"])
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Category: {category}\nQuestion: {question}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )


# ── MC (allenai/jeopardy_mc) ──────────────────────────────────────────────────

def jeopardy_mc_cf_prompt(line, task_name: str = None):
    """CF variant: cloze prompt, score each answer text."""
    category, question = _parse_context(line["context_original"])
    choices = line["choices"]["text"]
    labels = line["choices"]["label"]
    gold = labels.index(line["answerKey"])
    return Doc(
        task_name=task_name,
        query=f"Category: {category}\nQuestion: {question}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=gold,
    )


def jeopardy_mc_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options in prompt, score single label token."""
    category, question = _parse_context(line["context_original"])
    choices = line["choices"]["text"]
    labels = line["choices"]["label"]
    gold = labels.index(line["answerKey"])
    options = "\n".join(f" {l}. {t}" for l, t in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Category: {category}\nQuestion: {question}\n{options}\nAnswer:",
        choices=labels,
        gold_index=gold,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="jeopardy:bpb",
        prompt_function=jeopardy_bpb_prompt,
        hf_repo="soldni/jeopardy",
        hf_subset="mosaicml_gauntlet",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        stop_sequence=["\n\n", "Question:", "Category:"],
        metrics=[Metrics.target_bits_per_byte],
        version=0,
    ),
    LightevalTaskConfig(
        name="jeopardy:gen",
        prompt_function=jeopardy_gen_prompt,
        hf_repo="soldni/jeopardy",
        hf_subset="mosaicml_gauntlet",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=50,
        stop_sequence=["\n\n", "Question:", "Category:"],
        metrics=[Metrics.f1_score, Metrics.exact_match],
        version=0,
    ),
    LightevalTaskConfig(
        name="jeopardy_mc:cf",
        prompt_function=jeopardy_mc_cf_prompt,
        hf_repo="allenai/jeopardy_mc",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=["\n"],
        metrics=[
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            Metrics.target_bits_per_byte,
        ],
        version=0,
    ),
    LightevalTaskConfig(
        name="jeopardy_mc:mcf",
        prompt_function=jeopardy_mc_mcf_prompt,
        hf_repo="allenai/jeopardy_mc",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=["\n"],
        metrics=[
            Metrics.loglikelihood_acc,
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        ],
        version=0,
    ),
]
