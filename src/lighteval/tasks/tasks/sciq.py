"""
name:
Sciq

dataset:
allenai/sciq

abstract:
The SciQ dataset contains 13,679 crowdsourced science exam questions about
Physics, Chemistry and Biology, among others. The questions are in
multiple-choice format with 4 answer options each. For the majority of the
questions, an additional paragraph with supporting evidence for the correct
answer is provided.

languages:
english

tags:
physics, chemistry, biology, reasoning, multiple-choice, qa

paper:
https://arxiv.org/abs/1707.06209
"""

import hashlib
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
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


def sciq_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style, score full answer texts via logprobs."""
    return Doc(
        task_name=task_name,
        query=f"{line['support']}\nQuestion: {line['question']}\nAnswer:".strip(),
        choices=[
            f" {c}" for c in [line["distractor1"], line["distractor2"], line["distractor3"], line["correct_answer"]]
        ],
        gold_index=3,
    )


def sciq_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C/D options, score label tokens via logprobs."""
    gold_index = int(hashlib.md5(line["question"].encode()).hexdigest(), 16) % 4
    choices = [line["distractor1"], line["distractor2"], line["distractor3"]]
    choices.insert(gold_index, line["correct_answer"])

    query = "The following are multiple choice questions (with answers) about science.\n\n"
    query += f"Question: {line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, choices)])
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + letter for letter in ascii_uppercase[:len(choices)]],
        gold_index=gold_index,
        instruction="The following are multiple choice questions (with answers) about science.\n\n",
    )


def sciq_bpb_prompt(line, task_name: str = None):
    """BPB variant: CF-style prompt with gold correct_answer as single choice."""
    gold_text = line["correct_answer"]
    if not gold_text:
        return None
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )


# CF variant: completion-style, logprob on full answer text + BPB on gold choice
sciq_cf = LightevalTaskConfig(
    name="sciq:cf",
    prompt_function=sciq_cf_prompt,
    hf_repo="allenai/sciq",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
sciq_mcf = LightevalTaskConfig(
    name="sciq:mcf",
    prompt_function=sciq_mcf_prompt,
    hf_repo="allenai/sciq",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# Greedy variant: MCF-style prompt, generate 1 token, exact match
sciq_mcf_em = LightevalTaskConfig(
    name="sciq:mcf_em",
    prompt_function=sciq_mcf_prompt,
    hf_repo="allenai/sciq",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    sciq_cf,
    sciq_mcf,
    sciq_mcf_em,
]
