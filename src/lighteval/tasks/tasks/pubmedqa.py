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


def _context(line) -> str:
    """qiaojin/PubMedQA (the dataset this task loads) has context.contexts; the original JSON had CONTEXTS."""
    c = line["context"]["contexts"] if "context" in line else line["CONTEXTS"]
    return " ".join(c) if isinstance(c, list) else c


def _question(line) -> str:
    return line["question"] if "question" in line else line["QUESTION"]


def pubmedqa_cf_prompt(line, task_name: str = None):
    """CF variant: score full answer text (yes/no/maybe) via logprobs."""
    ctx = _context(line)
    return Doc(
        task_name=task_name,
        query=f"Abstract: {ctx}\nQuestion: {_question(line)}\nAnswer:",
        choices=[f" {c}" for c in _CHOICES],
        gold_index=_GOLD[line["final_decision"].lower()],
    )


def pubmedqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C options, score label tokens via logprobs."""
    ctx = _context(line)
    options = "\n".join(f" {l}. {c}" for l, c in zip("ABC", _CHOICES))
    return Doc(
        task_name=task_name,
        query=f"Abstract: {ctx}\nQuestion: {_question(line)}\n{options}\nAnswer:",
        choices=[" A", " B", " C"],
        gold_index=_GOLD[line["final_decision"].lower()],
    )


# qiaojin/PubMedQA: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"


TASKS_TABLE = [
    LightevalTaskConfig(
        name="pubmedqa:cf",
        prompt_function=pubmedqa_cf_prompt,
        hf_repo="qiaojin/PubMedQA",
        hf_revision=_REVISION,
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
        hf_revision=_REVISION,
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
        hf_revision=_REVISION,
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
