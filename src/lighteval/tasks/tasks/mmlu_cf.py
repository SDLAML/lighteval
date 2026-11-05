"""
name:
MMLU-CF

dataset:
microsoft/MMLU-CF

abstract:
A contamination-free multiple-choice benchmark, evaluated on its 14 public
category validation splits with the corresponding development splits for few-shots.
The category development splits currently share the same five examples.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2412.15194
"""

from functools import partial

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks import mmlu


# These are split prefixes in the default config, not Hugging Face subsets.
_MMLU_CF_CATEGORIES = (
    "Biology",
    "Business",
    "Chemistry",
    "Computer_Science",
    "Economics",
    "Engineering",
    "Health",
    "History",
    "Law",
    "Math",
    "Other",
    "Philosophy",
    "Physics",
    "Psychology",
)

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]
_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def mmlu_cf_prompt(line, task_name: str = None, *, subject: str, prompt_function):
    """Adapt the dataset's columns to the existing MMLU prompt formats."""
    return prompt_function(
        {
            "subject": subject,
            "question": line["Question"].strip(),
            "choices": [line[label].strip() for label in "ABCD"],
            "answer": line["Answer"].strip(),
        },
        task_name=task_name,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu_cf:{category.lower()}:{variant}",
        prompt_function=partial(
            mmlu_cf_prompt,
            subject=category.lower(),
            prompt_function=prompt_function,
        ),
        hf_repo="microsoft/MMLU-CF",
        hf_subset="default",
        hf_avail_splits=[f"{category}_val", f"{category}_dev"],
        evaluation_splits=[f"{category}_val"],
        few_shots_split=f"{category}_dev",
        few_shots_select=None,
        generation_size=generation_size,
        metrics=metrics,
        stop_sequence=stop_sequence,
        version=0,
    )
    for category in _MMLU_CF_CATEGORIES
    for variant, prompt_function, metrics, generation_size, stop_sequence in (
        ("cf", mmlu.mmlu_cf_prompt, _CF_METRICS, -1, ["\n"]),
        ("mcf", mmlu.mmlu_mcf_prompt, _MCF_METRICS, -1, ["\n"]),
        ("mcf_em", mmlu.mmlu_mcf_prompt, [Metrics.exact_match], 5, ["\n"]),
        ("chat", mmlu.mmlu_chat_prompt, [Metrics.gpqa_instruct_metric], 32768, ["Question:"]),
    )
]
