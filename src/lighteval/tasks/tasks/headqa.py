"""
name:
Headqa

dataset:
EleutherAI/headqa (the `headqa:cf|mcf|mcf_em` tasks, see the end of this file); lighteval/headqa_harness (the unused `_configs`)

abstract:
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to
access a specialized position in the Spanish healthcare system, and are
challenging even for highly specialized humans. They are designed by the
Ministerio de Sanidad, Consumo y Bienestar Social, who also provides direct
access to the exams of the last 5 years.

Version 2 of `headqa:cf|mcf|mcf_em` (owner 2026-11-13): the standard English TEST split only (2,742 questions) of EleutherAI/headqa
@ 3860ebd7 (config `en`, splits train 2,657 / validation 1,366 / test 2,742; HEAD-QA, Vilares and Gomez-Rodriguez, ACL 2019), few-shot
demonstrations from the VALIDATION split (never from test). Why validation and not train: every train question has 5 options (A-E),
every validation and test question has 4 (A-D); demonstrations in the format of the scored items cannot teach "E". Version 1 scored all
6,765 English questions (train + validation + test of the local titaneval parquet, which has no split column) and drew the shots from the
same 6,765. The split was verified row by row (the English rows of the old local file are EleutherAI's train + validation + test in that
order; the TitanEval-MCQ `headqa.parquet`, made by another loader from dvilares/head_qa, equals EleutherAI's test split row for row).

languages:
english, spanish

tags:
health, medical, multiple-choice, qa

paper:
https://arxiv.org/abs/1906.04701
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


def headqa_cf_prompt(line, task_name: str = None):
    """CF variant: score full answer texts via logprobs."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['qtext']}\nAnswer:",
        choices=[f" {answer['atext']}" for answer in line["answers"]],
        gold_index=int(line["ra"]) - 1,
    )


def headqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options, score label tokens via logprobs."""
    labels = list("ABCDE")[: len(line["answers"])]
    options = "\n".join(f" {l}. {a['atext']}" for l, a in zip(labels, line["answers"]))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['qtext']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=int(line["ra"]) - 1,
    )


def _configs(lang: str):
    return [
        LightevalTaskConfig(
            name=f"headqa:{lang}:cf",
            prompt_function=headqa_cf_prompt,
            hf_repo="lighteval/headqa_harness",
            hf_subset=lang,
            hf_avail_splits=["train", "test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling_from_train",
            generation_size=-1,
            metrics=_CF_METRICS,
            stop_sequence=["\n"],
            version=1,
        ),
        LightevalTaskConfig(
            name=f"headqa:{lang}:mcf",
            prompt_function=headqa_mcf_prompt,
            hf_repo="lighteval/headqa_harness",
            hf_subset=lang,
            hf_avail_splits=["train", "test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling_from_train",
            generation_size=-1,
            metrics=_MCF_METRICS,
            stop_sequence=["\n"],
            version=1,
        ),
        LightevalTaskConfig(
            name=f"headqa:{lang}:mcf_em",
            prompt_function=headqa_mcf_prompt,
            hf_repo="lighteval/headqa_harness",
            hf_subset=lang,
            hf_avail_splits=["train", "test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling_from_train",
            generation_size=1,
            metrics=[Metrics.exact_match],
            stop_sequence=["\n"],
            version=1,
        ),
    ]


# ── headqa: EleutherAI/headqa, English, pinned revision, scored on the test split ──
# (version 2; the local titaneval parquet `data/titaneval/headqa.parquet`, en + es, all splits, no split column, is no longer read)

REPO = "EleutherAI/headqa"
REVISION = "3860ebd7b109f45dfa97a9c8cb31b564165ca0f1"


def _choices(line) -> list[str]:
    return [a["atext"] for a in line["answers"]]


def _gold(line) -> int:
    """Position of the answer whose id is `ra`. No default: a row whose `ra` names no answer raises (it would score a wrong gold)."""
    aids = [a["aid"] for a in line["answers"]]
    return aids.index(line["ra"])


def _headqa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style, logprob on full answer text."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['qtext']}\nAnswer:",
        choices=[" " + c for c in _choices(line)],
        gold_index=_gold(line),
    )


def _headqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options, score label tokens via logprobs."""
    choices = _choices(line)
    labels = list("ABCDE")[: len(choices)]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['qtext']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=_gold(line),
    )


def _config(name: str, prompt_function, metrics, generation_size: int) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        prompt_function=prompt_function,
        hf_repo=REPO,
        hf_subset="en",
        hf_revision=REVISION,
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="validation",                   # 4-option questions like the test split (train has 5 options); never the scored split
        few_shots_select="random_sampling",             # 1 validation question equals a test question: the sampler skips a shot equal to the item
        generation_size=generation_size,
        metrics=metrics,
        stop_sequence=["\n"],
        version=2,
    )


headqa_cf = _config("headqa:cf", _headqa_cf_prompt, _CF_METRICS, -1)
headqa_mcf = _config("headqa:mcf", _headqa_mcf_prompt, _MCF_METRICS, -1)
headqa_mcf_em = _config("headqa:mcf_em", _headqa_mcf_prompt, [Metrics.exact_match], 1)


TASKS_TABLE = [
    headqa_cf,
    headqa_mcf,
    headqa_mcf_em,
]
