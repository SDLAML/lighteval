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
supergpqa:engineering:cf. The difficulty of each question is kept in the per-item
details (Doc.specific["difficulty"]), not as separate tasks.

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


_SUPERGPQA_COMMON = {
    "hf_repo": "m-a-p/SuperGPQA",
    "hf_subset": "default",
    # "hf_revision": "4430d4458112c7d4497fdcf94d7cc223313d6acf",
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
    # Only the 13 discipline tasks (all difficulties). The discipline x difficulty cells (`_SUPERGPQA_DIFFICULTIES`)
    # are NOT registered: they re-score the same questions (53,058 evaluations for 26,529 questions) and the
    # `supergpqa:<variant>` superset would average all 52 tasks, 1-question cells included. A difficulty breakdown
    # comes from `Doc.specific["difficulty"]` in the per-item details.
    for difficulty in [None]
]
