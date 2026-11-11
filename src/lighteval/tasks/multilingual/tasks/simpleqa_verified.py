"""
name:
SimpleQA Verified Multilingual

dataset:
ellamind/simpleqa-verified-multilingual

abstract:
Multilingual translations of SimpleQA Verified for short-form factual question
answering. Generative variants accept the canonical answer and its aliases;
BPB variants score only the canonical answer. MCF variants pair the canonical
answer with four easy or hard distractors, with reproducibly shuffled options.
MCF uses label log-probabilities; MCF EM generates the label and scores exact match.
Task names follow `simpleqa_verified:{language}:{gen,bpb}` and
`simpleqa_verified:{language}:{easy,hard}:{mcf,mcf_em}`. Use |0 for standard evaluation;
few-shot examples, when requested, are drawn from the eval split.

languages:
czech, danish, german, french, italian, dutch, polish, spanish

tags:
multilingual, factuality, general-knowledge, multiple-choice, qa

paper:
https://arxiv.org/abs/2509.07968
"""

import random
import re
from functools import partial

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm, remove_punc
from lighteval.metrics.utils.linguistic_tokenizers import get_word_tokenizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation, build_answers
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


_LANGUAGES = (
    Language.CZECH,
    Language.DANISH,
    Language.GERMAN,
    Language.FRENCH,
    Language.ITALIAN,
    Language.DUTCH,
    Language.POLISH,
    Language.SPANISH,
)


_NUMERIC_PART = re.compile(r"([+−-]?(?:\d+(?:[.,:/–−-]\d+)*|[.,]\d+)(?:\s*[%‰])?)")


def _normalize_answer(text: str, *, language: Language) -> str:
    # Preserve numeric separators and signs while removing surrounding punctuation.
    parts = _NUMERIC_PART.split(text.casefold())
    text = "".join(part if index % 2 else remove_punc(part) for index, part in enumerate(parts))
    # Articles can be valid short answers, such as French "un" or Italian "I".
    return " ".join(get_word_tokenizer(language).word_tokenize(text))


def _generative_metrics(language: Language):
    normalizer = partial(_normalize_answer, language=language)
    metrics = [
        MultilingualQuasiExactMatchMetric(language, "full"),
        MultilingualQuasiF1ScoreMetric(language),
    ]
    # Apply the same task-specific normalization to references and predictions.
    for metric in metrics:
        metric.sample_level_fn.normalize_gold = normalizer
        metric.sample_level_fn.normalize_pred = normalizer
    return metrics


def _adapter(line, *, include_aliases: bool):
    answer = line["answer"].strip()
    if not answer:
        return None
    choices = [answer]
    if include_aliases:
        choices.extend(alias.strip() for alias in (line.get("answer_aliases") or []) if alias.strip())
    # Keep the canonical answer first for deterministic few-shot examples.
    choices = list(dict.fromkeys(choices))
    return {
        "question": line["question"].strip(),
        "choices": choices,
        "gold_idx": list(range(len(choices))),
    }


def _get_qa_prompt_function(language: Language, *, include_aliases: bool):
    formulation = CFFormulation()
    base_prompt = get_mcq_prompt_function(language, adapter=lambda row: row, formulation=formulation)

    def prompt(line, task_name: str):
        adapted = _adapter(line, include_aliases=include_aliases)
        if adapted is None:
            return None
        doc = base_prompt(adapted, task_name)
        # The shared template capitalizes choices; preserve the original references.
        doc.choices = build_answers(adapted["choices"], formulation, TRANSLATION_LITERALS[language])
        return doc

    return prompt


def _mcf_adapter(line, *, difficulty: str):
    answer = line["answer"].strip()
    if not answer:
        return None
    choices = [answer, *(distractor.strip() for distractor in line[f"{difficulty}_distractors"])]
    order = list(range(len(choices)))
    # Stable row IDs keep option order aligned across translations and difficulties.
    random.Random(str(line.get("id") or line["question"])).shuffle(order)
    return {
        "question": line["question"].strip(),
        "choices": [choices[index] for index in order],
        "gold_idx": order.index(0),
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"simpleqa_verified:{language.value}:{variant}",
        prompt_function=_get_qa_prompt_function(
            language,
            include_aliases=variant == "gen",
        ),
        hf_repo="ellamind/simpleqa-verified-multilingual",
        hf_subset=language.value,
        hf_avail_splits=["eval"],
        evaluation_splits=["eval"],
        few_shots_split="eval",
        few_shots_select="random_sampling",
        generation_size=128 if variant == "gen" else -1,
        stop_sequence=["\n"],
        metrics=_generative_metrics(language) if variant == "gen" else [Metrics.target_bits_per_byte],
        version=1,
    )
    for language in _LANGUAGES
    for variant in ("gen", "bpb")
]


TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"simpleqa_verified:{language.value}:{difficulty}:{variant}",
        prompt_function=get_mcq_prompt_function(
            language,
            adapter=partial(_mcf_adapter, difficulty=difficulty),
            formulation=MCFFormulation(),
        ),
        hf_repo="ellamind/simpleqa-verified-multilingual",
        hf_subset=language.value,
        hf_avail_splits=["eval"],
        evaluation_splits=["eval"],
        few_shots_split="eval",
        few_shots_select="random_sampling",
        generation_size=5 if variant == "mcf_em" else -1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match]
        if variant == "mcf_em"
        else [LogLikelihoodAccMetric(), LogLikelihoodAccMetric(normalization=LogProbCharNorm())],
        version=0,
    )
    for language in _LANGUAGES
    for difficulty in ("easy", "hard")
    for variant in ("mcf", "mcf_em")
]
