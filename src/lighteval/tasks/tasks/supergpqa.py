"""
name:
SuperGPQA

dataset:
m-a-p/SuperGPQA

abstract:
SuperGPQA evaluates graduate-level knowledge across 285 disciplines using
multiple-choice questions with four to ten options. The full benchmark is
published in the train split; all variants evaluate that split. Subtasks filter
by discipline (13 tasks, all difficulties each), with names such as
supergpqa:engineering:cf; `supergpqa:mcf_em` is these 13 tasks. The difficulty of
each question is kept in the per-item details (Doc.specific["difficulty"]).
The letter-scoring generation task `mcf_em` also has the three difficulty tasks
`supergpqa:easy:mcf_em`, `supergpqa:middle:mcf_em` and `supergpqa:hard:mcf_em`
(7,731 / 11,748 / 7,050 questions): each is 13 cells `supergpqa:<discipline>:<difficulty>:mcf_em`
that evaluate the questions of one difficulty with exactly the prompts of the
discipline task (see `_difficulty_cell`).

languages:
english

tags:
graduate-level, knowledge, multiple-choice, qa, reasoning

paper:
https://arxiv.org/abs/2502.14739
"""

from dataclasses import replace
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


_SUPERGPQA_DISCIPLINES = [
    "Agronomy",
    "Economics",
    "Education",
    "Engineering",
    "History",
    "Law",
    "Literature and Arts",
    "Management",
    "Medicine",
    "Military Science",
    "Philosophy",
    "Science",
    "Sociology",
]
_SUPERGPQA_DIFFICULTIES = ["easy", "middle", "hard"]


def _matches_subset(line, discipline: str, difficulty: str | None = None):
    return line["discipline"] == discipline and (difficulty is None or line["difficulty"] == difficulty)


def _matches_sample_subset(sample: Sample, discipline: str, difficulty: str | None = None):
    return _matches_subset(sample.metadata, discipline, difficulty)


def supergpqa_cf_prompt(line, task_name: str = None):
    """Score full answer texts as continuations of the question."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question'].strip()}\nAnswer:",
        choices=[option if option and option[0].isspace() else f" {option}" for option in line["options"]],
        gold_index=ascii_uppercase.index(line["answer_letter"]),
    )


def _doc_has_difficulty(doc, difficulty: str):
    return doc.specific["difficulty"] == difficulty


def supergpqa_mcf_prompt(line, task_name: str = None):
    """Show all options and score their letter labels."""
    labels = ascii_uppercase[: len(line["options"])]
    options = "\n".join(f"{label}: {option}" for label, option in zip(labels, line["options"]))
    return Doc(
        task_name=task_name,
        query=f"Answer the following multiple choice question.\n\nQuestion: {line['question'].strip()}\n{options}\nAnswer:",
        choices=[f" {label}" for label in labels],
        gold_index=labels.index(line["answer_letter"]),
        # kept in the per-item details (Doc), so a breakdown by difficulty / discipline can be computed offline
        specific={"uuid": line["uuid"], "discipline": line["discipline"], "difficulty": line["difficulty"]},
    )


def supergpqa_cot_prompt(line, task_name: str = None):
    """Request reasoning followed by a final answer letter."""
    labels = ascii_uppercase[: len(line["options"])]
    instruction = (
        "Answer the following multiple choice question. The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
        f"{labels}. Think step by step before answering."
    )
    options = "\n".join(f"{label}: {option}" for label, option in zip(labels, line["options"]))
    return Doc(
        task_name=task_name,
        query=f"{instruction}\n\n{line['question'].strip()}\n\n{options}\n\nAnswer:",
        choices=[f" {label}" for label in labels],
        gold_index=labels.index(line["answer_letter"]),
        instruction=instruction,
    )


def record_to_sample(record):
    return Sample(
        input=record["question"].strip(),
        choices=record["options"],
        target=record["answer_letter"],
        metadata={"discipline": record["discipline"], "difficulty": record["difficulty"]},
    )


def sample_to_fewshot(sample):
    options = "\n".join(f"{label}) {option}" for label, option in zip(ascii_uppercase, sample.choices))
    return f"{sample.input}\n\n{options}\n\nANSWER: {sample.target}"


# m-a-p/SuperGPQA: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "4430d4458112c7d4497fdcf94d7cc223313d6acf"


_SUPERGPQA_COMMON = {
    "hf_repo": "m-a-p/SuperGPQA",
    "hf_revision": _REVISION,
    "hf_subset": "default",
    "hf_avail_splits": ["train"],
    "evaluation_splits": ["train"],
    "few_shots_split": "train",
    "few_shots_select": "random_sampling",
    "sample_fields": record_to_sample,
    "sample_to_fewshot": sample_to_fewshot,
    "version": 0,
}

supergpqa_cf = LightevalTaskConfig(
    name="supergpqa:cf",
    prompt_function=supergpqa_cf_prompt,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    generation_size=-1,
    metrics=[
        LogLikelihoodAccMetric(),
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        Metrics.target_bits_per_byte,
    ],
    stop_sequence=["\n"],
    **_SUPERGPQA_COMMON,
)

supergpqa_mcf = LightevalTaskConfig(
    name="supergpqa:mcf",
    prompt_function=supergpqa_mcf_prompt,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    generation_size=-1,
    metrics=[
        LogLikelihoodAccMetric(),
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    ],
    stop_sequence=["\n"],
    **_SUPERGPQA_COMMON,
)

supergpqa_mcf_em = LightevalTaskConfig(
    name="supergpqa:mcf_em",
    prompt_function=supergpqa_mcf_prompt,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    **_SUPERGPQA_COMMON,
)

supergpqa_cot = LightevalTaskConfig(
    name="supergpqa:cot",
    prompt_function=supergpqa_cot_prompt,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    generation_size=32768,
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],
    **_SUPERGPQA_COMMON,
)


def _task_slug(discipline: str) -> str:
    return discipline.lower().replace(" ", "_")


def _difficulty_cell(discipline: str, difficulty: str) -> LightevalTaskConfig:
    """`supergpqa:<discipline>:<difficulty>:mcf_em` (owner 11-16: three tasks easy / middle / hard). The cell evaluates the questions of ONE difficulty of one discipline with
    EXACTLY the prompt they have in the discipline task `supergpqa:<discipline>:mcf_em`: the same `hf_filter` (so the same few-shot pool = the whole discipline, not the cell), the
    same shuffle and random stream, and the difficulty is applied afterwards (`eval_doc_filter`: every question of the discipline is drawn for, only the cell's are scored). It is in
    the supersets `supergpqa:<difficulty>:mcf_em` only (`superset_keys`), so `supergpqa:mcf_em` stays the 13 discipline tasks and no question is evaluated twice by one name."""
    return replace(
        supergpqa_mcf_em,
        name=f"supergpqa:{_task_slug(discipline)}:{difficulty}:mcf_em",
        hf_filter=partial(_matches_subset, discipline=discipline, difficulty=None),
        filter=partial(_matches_sample_subset, discipline=discipline, difficulty=difficulty),
        eval_doc_filter=partial(_doc_has_difficulty, difficulty=difficulty),
        superset_keys=(f"supergpqa:{difficulty}:mcf_em",),
    )


# Keep the variant settings shared across discipline and difficulty subtasks.
TASKS_TABLE = [
    replace(
        config,
        name=(
            f"supergpqa:{discipline.lower().replace(' ', '_')}"
            f"{':' + difficulty if difficulty else ''}:{config.name.rsplit(':', 1)[1]}"
        ),
        hf_filter=partial(_matches_subset, discipline=discipline, difficulty=difficulty),
        filter=partial(_matches_sample_subset, discipline=discipline, difficulty=difficulty),
    )
    for config in [supergpqa_cf, supergpqa_mcf, supergpqa_mcf_em, supergpqa_cot]
    for discipline in _SUPERGPQA_DISCIPLINES
    # The 13 discipline tasks (all difficulties) of every variant. The discipline x difficulty cells of the other variants
    # are NOT registered: they re-score the same questions (53,058 evaluations for 26,529 questions) and the
    # `supergpqa:<variant>` superset would average all 52 tasks, 1-question cells included. A difficulty breakdown
    # comes from `Doc.specific["difficulty"]` in the per-item details.
    for difficulty in [None]
] + [
    # `mcf_em` only (owner 11-16): the 39 cells of the three difficulty tasks (supersets supergpqa:easy|middle|hard:mcf_em, 13 cells each; they are in no other superset).
    _difficulty_cell(discipline, difficulty)
    for difficulty in _SUPERGPQA_DIFFICULTIES
    for discipline in _SUPERGPQA_DISCIPLINES
]
