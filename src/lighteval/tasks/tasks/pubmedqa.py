"""
name:
Pubmedqa

dataset:
qiaojin/PubMedQA

abstract:
PubMedQA is a dataset for biomedical research question answering.
Each question is answerable with yes, no, or maybe based on a PubMed abstract.

languages:
english

tags:
biomedical, health, medical, qa

paper:
https://pubmedqa.github.io/
"""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_CHOICES = ["yes", "no", "maybe"]
_GOLD = {c: i for i, c in enumerate(_CHOICES)}

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def pubmedqa_cf_prompt(line, task_name: str = None):
    """CF variant: score full answer text (yes/no/maybe) via logprobs."""
    ctx = " ".join(line["CONTEXTS"]) if isinstance(line["CONTEXTS"], list) else line["CONTEXTS"]
    return Doc(
        task_name=task_name,
        query=f"Abstract: {ctx}\nQuestion: {line['QUESTION']}\nAnswer:",
        choices=[f" {c}" for c in _CHOICES],
        gold_index=_GOLD[line["final_decision"].lower()],
    )


def pubmedqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C options, score label tokens via logprobs."""
    ctx = " ".join(line["CONTEXTS"]) if isinstance(line["CONTEXTS"], list) else line["CONTEXTS"]
    options = "\n".join(f" {l}. {c}" for l, c in zip("ABC", _CHOICES))
    return Doc(
        task_name=task_name,
        query=f"Abstract: {ctx}\nQuestion: {line['QUESTION']}\n{options}\nAnswer:",
        choices=[" A", " B", " C"],
        gold_index=_GOLD[line["final_decision"].lower()],
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="pubmedqa:cf",
        prompt_function=pubmedqa_cf_prompt,
        hf_repo="qiaojin/PubMedQA",
        hf_subset="pqa_labeled",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=-1,
        metrics=_CF_METRICS,
        stop_sequence=["\n"],
        version=1,
    ),
    LightevalTaskConfig(
        name="pubmedqa:mcf",
        prompt_function=pubmedqa_mcf_prompt,
        hf_repo="qiaojin/PubMedQA",
        hf_subset="pqa_labeled",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=1,
    ),
    LightevalTaskConfig(
        name="pubmedqa:mcf_em",
        prompt_function=pubmedqa_mcf_prompt,
        hf_repo="qiaojin/PubMedQA",
        hf_subset="pqa_labeled",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=1,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=1,
    ),
]
