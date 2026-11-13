"""
name:
R-Bench (text only)

dataset:
R-Bench/R-Bench

abstract:
Graduate-level multidisciplinary reasoning questions in English and Chinese.
Only rbench-t_en and rbench-t_zh are included, with six answer options per
question. Rows without a single valid answer letter are excluded: the current
release labels question 236 as "E\n\nF" in both languages.

languages:
english, chinese

tags:
graduate-level, multiple-choice, qa, reasoning

paper:
https://arxiv.org/abs/2505.02018
"""

from functools import partial

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


_LABELS = tuple("ABCDEF")
_PROMPT_LABELS = {"en": ("Question:", "Answer:"), "zh": ("问题：", "答案：")}
_COT_INSTRUCTIONS = {
    "en": (
        "Answer the following multiple choice question. Think step by step before answering. "
        "The last line of your response should be 'Answer: $LETTER' (without quotes), "
        "where LETTER is one of ABCDEF."
    ),
    "zh": (
        "请回答以下选择题，并逐步推理。回答的最后一行应为 'Answer: $LETTER'（不含引号），"
        "其中 LETTER 为 ABCDEF 中的一个字母。"
    ),
}
_INSPECT_TEMPLATES = {
    "en": "Answer the following multiple choice question.\n\n{question}\n\n{choices}\n\n",
    "zh": "请回答以下选择题。\n\n{question}\n\n{choices}\n\n",
}


def rbench_filter(line):
    """Exclude ambiguous annotations instead of guessing a single gold answer."""
    return line["answer"].strip() in _LABELS


def _sample_filter(sample):
    return sample.target in _LABELS


def rbench_prompt(line, task_name: str = None, *, language: str = "en", formulation: str = "mcf"):
    question_label, answer_label = _PROMPT_LABELS[language]
    question = line["question"].strip()
    options = [line[label] for label in _LABELS]
    instruction = ""
    if formulation == "cf":
        query = f"{question_label} {question}\n{answer_label}"
        choices = [option if option[0].isspace() else f" {option}" for option in options]
    else:
        option_text = "\n".join(f"{label}: {option}" for label, option in zip(_LABELS, options))
        query = f"{question_label} {question}\n{option_text}\n{answer_label}"
        choices = [f" {label}" for label in _LABELS]
        if formulation == "cot":
            instruction = _COT_INSTRUCTIONS[language]
            query = f"{instruction}\n\n{question}\n\n{option_text}"
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=_LABELS.index(line["answer"].strip()),
        instruction=instruction,
    )


def record_to_sample(record):
    return Sample(
        id=record["index"],
        input=record["question"].strip(),
        choices=[record[label] for label in _LABELS],
        target=record["answer"].strip(),
    )


def sample_to_fewshot(sample):
    options = "\n".join(f"{label}) {option}" for label, option in zip(_LABELS, sample.choices))
    return f"{sample.input}\n\n{options}\n\nANSWER: {sample.target}"


# R-Bench/R-Bench: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "c0e92a6d90dad980ea51a47af64c12ec67b670ce"


def _configs(language):
    common = {
        "hf_repo": "R-Bench/R-Bench",
        "hf_revision": _REVISION,
        "hf_subset": f"rbench-t_{language}",
        "hf_avail_splits": ["test"],
        "evaluation_splits": ["test"],
        # There is no dedicated few-shot split; use |0 for standard evaluation.
        "few_shots_split": "test",
        "few_shots_select": "random_sampling",
        "hf_filter": rbench_filter,
        "sample_fields": record_to_sample,
        "sample_to_fewshot": sample_to_fewshot,
        "filter": _sample_filter,
        "version": 0,
    }
    likelihood_metrics = [LogLikelihoodAccMetric(), LogLikelihoodAccMetric(normalization=LogProbCharNorm())]
    configs = []
    for variant, metrics, generation_size in [
        ("cf", [*likelihood_metrics, Metrics.target_bits_per_byte], -1),
        ("mcf", likelihood_metrics, -1),
        ("mcf_em", [Metrics.exact_match], 5),
        ("cot", [Metrics.gpqa_instruct_metric], 32768),
    ]:
        template = _INSPECT_TEMPLATES[language]
        if variant == "cot":
            template += _COT_INSTRUCTIONS[language].replace("Answer:", "ANSWER:")
        else:
            template += (
                "Reply with 'ANSWER: $LETTER' (without quotes), where LETTER is the correct option."
                if language == "en"
                else "请按 'ANSWER: $LETTER'（不含引号）的格式回答，其中 LETTER 为正确选项的字母。"
            )
        configs.append(
            LightevalTaskConfig(
                name=f"rbench:{language}:{variant}",
                prompt_function=partial(rbench_prompt, language=language, formulation=variant),
                solver=[multiple_choice(template=template, cot=variant == "cot", cache=True)],
                scorer=choice(),
                generation_size=generation_size,
                metrics=metrics,
                stop_sequence=[] if variant == "cot" else ["\n"],
                **common,
            )
        )
    return configs


TASKS_TABLE = [config for language in ("en", "zh") for config in _configs(language)]
