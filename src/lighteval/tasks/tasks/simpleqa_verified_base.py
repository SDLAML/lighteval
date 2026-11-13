"""
name:
SimpleQA Verified Multilingual, base-model protocol (short-answer generation)

dataset:
ellamind/simpleqa-verified-multilingual

abstract:
Machine translations (Gemini 3.1 Pro, by ellamind) of the 1,000 SimpleQA Verified questions into Czech, Danish, German, French,
Italian, Dutch, Polish and Spanish: NON-ENGLISH ONLY (no English split exists in this dataset). Short-form factual recall in the
format used for base-model checkpoints: the language's own "question / answer" cloze prompt of Oleg Filatov's
simpleqa_verified:<lang>:gen (simpleqa_verified.py), five fixed demonstrations, at most 128 new tokens, stop at newline, and the
fork's quasi exact match / quasi F1 against the canonical answer and its aliases (no judge). The questions of the six small
languages are the same 100 questions (ces, dan, ita, nld, pol, spa 100 rows; fra 300; deu 1,000; the card says fra 100).
The five demonstrations are 5 of the 100 questions that exist in all 8 languages (one per answer type where possible, fixed sha256
rule, heldout_shots.py), held out of every language and not scored: 95 / 95 / 995 / 295 / 95 / 95 / 95 / 95 questions scored.
The metrics are named em / f1 (the fork's names carry the language, which breaks the macro mean over languages).
simpleqa_verified:* in the multilingual folder is unchanged (its few-shot examples come from the scored split).

languages:
czech, danish, german, french, italian, dutch, polish, spanish

tags:
multilingual, factuality, general-knowledge, qa

paper:
https://arxiv.org/abs/2509.07968
"""

from functools import partial

from lighteval.metrics.dynamic_metrics import MultilingualQuasiExactMatchMetric, MultilingualQuasiF1ScoreMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.simpleqa_verified import (
    _LANGUAGES,
    _generative_metrics,
    _get_qa_prompt_function,
)
from lighteval.tasks.tasks.heldout_shots import rows_by_key


REPO = "ellamind/simpleqa-verified-multilingual"
REVISION = "5975dec960b4d273708d58888ea23a020265f619"  # main on 2026-11-13 (card: license CC-BY-4.0)
# Held-out demonstrations: ids present in all 8 languages, in the order shown. Rule: select_shots over the 100 ids that exist in
# every language with answer_type as label (distinct types first), unit-tested on the pinned dataset.
SHOT_IDS = (
    "simpleqa_verified_1230",
    "simpleqa_verified_1181",
    "simpleqa_verified_1315",
    "simpleqa_verified_1202",
    "simpleqa_verified_1081",
)


def _id_of(row: dict) -> str:
    return row["id"]


def _prompt_function(language):
    base = _get_qa_prompt_function(language, include_aliases=True)

    def prompt(line, task_name: str = None):
        if not line.get("__few_shots") and line["id"] in SHOT_IDS:
            return None  # held out: a demonstration only
        return base(line, task_name)

    return prompt


def _metrics(language):
    # em / f1 with the fork's normaliser and scorers (alias-aware: max over the canonical answer and its aliases)
    metrics = _generative_metrics(language)
    for metric in metrics:
        if isinstance(metric, MultilingualQuasiExactMatchMetric):
            metric.metric_name = "em"
        elif isinstance(metric, MultilingualQuasiF1ScoreMetric):
            metric.metric_name = "f1"
    return metrics


def shot_rows(language: str) -> list[dict]:
    return rows_by_key(REPO, language, "eval", REVISION, _id_of, SHOT_IDS)


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"simpleqa_verified_base:{language.value}:gen",
        prompt_function=_prompt_function(language),
        hf_repo=REPO,
        hf_subset=language.value,
        hf_revision=REVISION,
        hf_avail_splits=["eval"],
        evaluation_splits=["eval"],
        few_shots_rows=partial(shot_rows, language.value),
        few_shots_select="sequential",
        generation_size=128,
        stop_sequence=["\n"],
        metrics=_metrics(language),
        version=0,
    )
    for language in _LANGUAGES
]
