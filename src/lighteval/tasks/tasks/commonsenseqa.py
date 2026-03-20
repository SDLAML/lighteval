"""
name:
Commonsenseqa

dataset:
tau/commonsense_qa

abstract:
CommonsenseQA is a new multiple-choice question answering dataset that requires
different types of commonsense knowledge to predict the correct answers . It
contains 12,102 questions with one correct answer and four distractor answers.
The dataset is provided in two major training/validation/testing set splits:
"Random split" which is the main evaluation split, and "Question token split",
see paper for details.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
https://arxiv.org/abs/1811.00937
"""

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


def commonsenseqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options in prompt, score label tokens via logprobs."""
    choices = line["choices"]["text"]
    options = "\n".join(f" {l}. {c}" for l, c in zip(ascii_uppercase[:len(choices)], choices))
    query = f"Question: {line['question']}\n{options}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + l for l in ascii_uppercase[:len(choices)]],
        gold_index=list(ascii_uppercase).index(line["answerKey"].strip()),
    )


def commonsenseqa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt with full answer texts."""
    query = f"Question: {line['question']}\nAnswer:"
    gold_ix = list(ascii_uppercase).index(line["answerKey"].strip())
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + c for c in line["choices"]["text"]],
        gold_index=gold_ix,
    )



# Greedy variant: MCF-style prompt, generate 1 token, exact match
commonsenseqa_mcf_em = LightevalTaskConfig(
    name="commonsenseqa:mcf_em",
    prompt_function=commonsenseqa_mcf_prompt,
    hf_repo="tau/commonsense_qa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
commonsenseqa_mcf = LightevalTaskConfig(
    name="commonsenseqa:mcf",
    prompt_function=commonsenseqa_mcf_prompt,
    hf_repo="tau/commonsense_qa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# CF variant: completion-style, logprob on full answer text + BPB on gold choice
commonsenseqa_cf = LightevalTaskConfig(
    name="commonsenseqa:cf",
    prompt_function=commonsenseqa_cf_prompt,
    hf_repo="tau/commonsense_qa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    commonsenseqa_mcf_em,
    commonsenseqa_mcf,
    commonsenseqa_cf,
]
