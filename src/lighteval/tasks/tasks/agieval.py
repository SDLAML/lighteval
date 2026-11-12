"""
name:
Agieval

dataset:
lighteval/agi_eval_en

abstract:
AGIEval is a human-centric benchmark specifically designed to evaluate the
general abilities of foundation models in tasks pertinent to human cognition and
problem-solving. This benchmark is derived from 20 official, public, and
high-standard admission and qualification exams intended for general human
test-takers, such as general college admission tests (e.g., Chinese College
Entrance Exam (Gaokao) and American SAT), law school admission tests, math
competitions, lawyer qualification tests, and national civil service exams.

languages:
english

tags:
biology, chemistry, geography, history, knowledge, language, multiple-choice, physics, reasoning

paper:
https://arxiv.org/abs/2304.06364
"""

import re
from functools import partial
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.agieval_demos import AGIEVAL_DEMOS


_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def record_to_sample(record):
    """Read the English AGIEval schema, preserving passage context and option text."""
    labels = ascii_uppercase[: len(record["options"])]
    if not labels or len(record["options"]) > len(ascii_uppercase) or record["label"] not in list(labels):
        raise ValueError("AGIEval multiple-choice records must have options and a valid answer letter.")

    options = record["options"]
    option_labels = []
    for option in options:
        match = re.match(r"^\s*(?:\(([A-Z])\)|([A-Z])(?:[.)]|\s+))", option)
        option_labels.append((match[1] or match[2]) if match else None)
    if all(option_labels):
        if set(option_labels) != set(labels):
            raise ValueError("AGIEval option labels must be unique and cover the available choices.")
        # Some LogiQA rows store B and C out of order. Golds refer to these
        # original option labels, so restore label order before relabeling.
        options = [option for _, option in sorted(zip(option_labels, options))]

    choices = [
        re.sub(rf"^\s*(?:\({label}\)\s*|{label}(?:[.)]\s*|\s+))", "", option, count=1).strip()
        for label, option in zip(labels, options)
    ]
    question = f"Question: {record['question'].strip()}"
    passage = (record.get("passage") or "").strip()
    if passage:
        question = f"Passage:\n{passage}\n\n{question}"
    return Sample(input=question, choices=choices, target=record["label"])


def sample_to_fewshot(sample):
    options = "\n".join(f"{label}. {option}" for label, option in zip(ascii_uppercase, sample.choices))
    return f"{sample.input}\n\n{options}\n\nAnswer: {sample.target}"


def _is_valid_sample(sample: Sample) -> bool:
    # SAT-English train rows 100 and 101 have identical content but golds B and
    # A. Exclude both, matching content so shuffling cannot change the filter.
    return not (
        sample.input.startswith(
            "Passage:\nAt last, Old Widow Lau was done haggling with the driver "
            "and we stepped inside Father's shop."
        )
        and sample.input.endswith(
            "\n\nQuestion: Throughout the passage, the narrator is portrayed as someone who is"
        )
        and sample.choices == [
            "reserved around unfamiliar people.",
            "attuned to her immediate surroundings.",
            "sympathetic to the needs of others.",
            "anxious about her responsibilities.",
        ]
    )


def agieval_prompt_cot_eng(line, task_name: str = None):
    sample = record_to_sample(line)
    if not _is_valid_sample(sample):
        return None
    labels = ascii_uppercase[: len(sample.choices)]
    instruction = (
        "Answer the following multiple choice question. The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
        f"{labels}. Think step by step before answering."
    )
    options = "\n".join(f"{label}. {option}" for label, option in zip(labels, sample.choices))
    query = f"{instruction}\n\n{sample.input}\n\n{options}\n\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {label}" for label in labels],
        gold_index=labels.index(sample.target),
        instruction=instruction,
    )


def agieval_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options in prompt, score label tokens via logprobs."""
    sample = record_to_sample(line)
    if not _is_valid_sample(sample):
        return None
    labels = ascii_uppercase[: len(sample.choices)]
    instruction = "The following are multiple choice questions (with answers).\n\n"
    options = "\n".join(f"{label}. {option}" for label, option in zip(labels, sample.choices))
    gold_index = labels.index(sample.target)
    return Doc(
        task_name=task_name,
        query=f"{instruction}{sample.input}\n\n{options}\n\nAnswer:",
        choices=[f" {label}" for label in labels],
        gold_index=gold_index,
        fewshot_sorting_class=sample.choices[gold_index],
        instruction=instruction,
    )


AGIEVAL_SUBSETS = [
    "aqua_rat",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
    "sat-math",
]

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"agieval_eng:{subset}:cot",
        sample_fields=record_to_sample,
        filter=_is_valid_sample,
        sample_to_fewshot=sample_to_fewshot,
        solver=[multiple_choice(cot=True, max_tokens=32768)],
        scorer=choice(),
        prompt_function=agieval_prompt_cot_eng,
        hf_repo="lighteval/agi_eval_en",
        hf_subset=subset,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["train"],
        # The Hub's LogiQA validation split contains AQUA questions. Sample genuine
        # LogiQA examples instead; the few-shot sampler excludes the evaluated row.
        few_shots_split="train" if subset == "logiqa-en" else "validation",
        few_shots_select="random_sampling",
        generation_size=32768,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Question:"],
        version=4,
    )
    for subset in AGIEVAL_SUBSETS
]

def _agieval_demo_rows(subset: str) -> list[dict]:
    """The official AGIEval demos of a subset (3 or 5), in the official order (agieval_demos.py)."""
    return AGIEVAL_DEMOS[subset.replace("_", "-")]


# The mcf / mcf_em variants take their shots ONLY from the official demos: the first K of them, in order, K =
# min(requested K, official count) (3 for lsat-ar, lsat-lr, lsat-rc, logiqa-en, sat-en; 5 for sat-math, aqua_rat).
# They are never sampled from the evaluated split. A smaller pool than K logs a warning (prompt_manager.py) and the
# `n_shots` metric / the INFO line of `get_docs` give the shots really used. A demo identical to the evaluated item
# would be excluded (none is: checked passage + question + options).
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"agieval_eng:{subset}:mcf",
        prompt_function=agieval_mcf_prompt,
        hf_repo="lighteval/agi_eval_en",
        hf_subset=subset,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["train"],
        few_shots_rows=partial(_agieval_demo_rows, subset),
        few_shots_select="sequential",
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=1,
    )
    for subset in AGIEVAL_SUBSETS
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"agieval_eng:{subset}:mcf_em",
        prompt_function=agieval_mcf_prompt,
        hf_repo="lighteval/agi_eval_en",
        hf_subset=subset,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["train"],
        few_shots_rows=partial(_agieval_demo_rows, subset),
        few_shots_select="sequential",
        generation_size=5,
        metrics=[Metrics.exact_match, Metrics.n_shots],
        stop_sequence=["\n"],
        version=2,
    )
    for subset in AGIEVAL_SUBSETS
]
