"""
name:
Bigbench Hard

dataset:
lukaemon/bbh

abstract:
BIG-Bench Hard (BBH) is a suite of 27 challenging BIG-Bench tasks requiring
multi-step reasoning. We evaluate with 3-shot chain-of-thought prompting.

languages:
english

tags:
reasoning

paper:
https://arxiv.org/abs/2210.09261
"""

import re

import numpy as np

from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


_BBH_SUBSETS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

_ANS_RE = re.compile(r"(?i)the answer is:?\s*(.+?)(?:[.\n]|$)")


class _BBHCoTExactMatch:
    """Extract the answer after 'the answer is' from CoT output, then exact-match against gold."""

    def __call__(self, predictions, formatted_doc, **kwargs):
        pred = predictions[0].strip() if predictions else ""
        gold = formatted_doc.choices[formatted_doc.gold_index].strip()
        match = _ANS_RE.search(pred)
        if match:
            extracted = match.group(1).strip().rstrip(".,")
        else:
            extracted = pred.strip()
        return int(extracted.lower() == gold.lower())


bbh_cot_exact_match = SampleLevelMetric(
    metric_name="em",
    sample_level_fn=_BBHCoTExactMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def bbh_cot_prompt(line, task_name: str = None):
    is_few_shots = line.get("__few_shots", False)
    query = f"Q: {line['input']}\nA: Let's think step by step."
    if is_few_shots:
        # Few-shot exemplar: provide the full CoT answer
        return Doc(
            task_name=task_name,
            query=query,
            choices=[f" {line['target']}"],
            gold_index=0,
        )
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["target"]],
        gold_index=0,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"bigbench_hard:{subset}",
        prompt_function=bbh_cot_prompt,
        hf_repo="lukaemon/bbh",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling_from_train",
        generation_size=1024,
        metrics=[bbh_cot_exact_match],
        stop_sequence=["</s>", "Q", "\n\n"],
        version=0,
    )
    for subset in _BBH_SUBSETS
]
