"""
name:
Med

dataset:
lighteval/med_mcqa, lighteval/med_paragraph_simplification, bigbio/med_qa

abstract:
A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering

languages:
english

tags:
health, medical

paper:
https://medmcqa.github.io/
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


def med_mcqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C/D options, score label tokens via logprobs.

    Matches OLMO's MedMCQAMC: plain Question + options, no instruction prefix.
    """
    query = f"Question: {line['question']}\n"
    query += "".join(
        [
            f" {key}. {choice}\n"
            for key, choice in zip(ascii_uppercase, [line["opa"], line["opb"], line["opc"], line["opd"]])
        ]
    )
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + c for c in list(ascii_uppercase)[:4]],
        gold_index=line["cop"] - 1,
    )


def med_paragraph_simplification_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"###\nArticle:{line['query']}\n\nSummarize the above article in 10 sentences.\n",
        gold_index=0,
        choices=[line["answer"]],
    )


def med_qa_prompt(line, task_name: str = None):
    """MCF variant: labeled options, score label tokens via logprobs."""
    query = f"Give a letter answer among A, B, C or D.\nQuestion: {line['question']}\n"
    query += "".join([f"{option['key']}. {option['value']}\n" for option in line["options"]])
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + opt["key"] for opt in line["options"]],
        gold_index=list(ascii_uppercase).index(line["answer_idx"]),
        instruction="Give a letter answer among A, B, C or D.\n",
    )


def med_qa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt with full answer texts as choices."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + opt["value"] for opt in line["options"]],
        gold_index=list(ascii_uppercase).index(line["answer_idx"]),
    )


def med_qa_mcf_em_prompt(line, task_name: str = None):
    """Greedy variant: generate 1 token, compare with gold letter."""
    query = f"Give a letter answer among A, B, C or D.\nQuestion: {line['question']}\n"
    query += "".join([f"{option['key']}. {option['value']}\n" for option in line["options"]])
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[opt["key"] for opt in line["options"]],
        gold_index=list(ascii_uppercase).index(line["answer_idx"]),
        instruction="Give a letter answer among A, B, C or D.\n",
    )


def med_mcqa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt with full answer texts as choices."""
    answers = [line["opa"], line["opb"], line["opc"], line["opd"]]
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + a for a in answers],
        gold_index=line["cop"] - 1,
    )


# Greedy variant: MCF-style prompt, generate token(s), exact match
med_mcqa_mcf_em = LightevalTaskConfig(
    name="med_mcqa:mcf_em",
    prompt_function=med_mcqa_mcf_prompt,
    hf_repo="lighteval/med_mcqa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
med_mcqa_mcf = LightevalTaskConfig(
    name="med_mcqa:mcf",
    prompt_function=med_mcqa_mcf_prompt,
    hf_repo="lighteval/med_mcqa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# MCF logprob variant: labeled options, score label tokens via logprobs
med_qa_mcf = LightevalTaskConfig(
    name="med_qa:mcf",
    prompt_function=med_qa_prompt,
    hf_repo="bigbio/med_qa",
    hf_subset="med_qa_en_source",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# Greedy variant: generate 1 token, exact match
med_qa_mcf_em = LightevalTaskConfig(
    name="med_qa:mcf_em",
    prompt_function=med_qa_mcf_em_prompt,
    hf_repo="bigbio/med_qa",
    hf_subset="med_qa_en_source",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

# CF variant: completion-style, logprob on full answer text + BPB on gold choice
med_mcqa_cf = LightevalTaskConfig(
    name="med_mcqa:cf",
    prompt_function=med_mcqa_cf_prompt,
    hf_repo="lighteval/med_mcqa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# ── med_qa: titaneval_local (cached parquet) ──

def _med_qa_titaneval_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style, logprob on full answer text."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in line["choices"]],
        gold_index=line["answer_index"],
    )


def _med_qa_titaneval_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C/D options, score label tokens via logprobs."""
    labels = list(ascii_uppercase)[: len(line["choices"])]
    options = "\n".join(f" {l}. {c}" for l, c in zip(labels, line["choices"]))
    return Doc(
        task_name=task_name,
        query=f"Give a letter answer among A, B, C or D.\nQuestion: {line['question']}\n{options}\nAnswer:",
        choices=[" " + l for l in labels],
        gold_index=line["answer_index"],
        instruction="Give a letter answer among A, B, C or D.\n",
    )


med_qa_titaneval_cf = LightevalTaskConfig(
    name="med_qa:cf",
    prompt_function=_med_qa_titaneval_cf_prompt,
    hf_repo="titaneval_local",
    hf_subset="med_qa",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=1,
)

med_qa_titaneval_mcf = LightevalTaskConfig(
    name="med_qa:mcf",
    prompt_function=_med_qa_titaneval_mcf_prompt,
    hf_repo="titaneval_local",
    hf_subset="med_qa",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=1,
)

med_qa_titaneval_mcf_em = LightevalTaskConfig(
    name="med_qa:mcf_em",
    prompt_function=_med_qa_titaneval_mcf_prompt,
    hf_repo="titaneval_local",
    hf_subset="med_qa",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

# ── original bigbio/med_qa configs (broken: script-based dataset) ──
# med_qa_cf = LightevalTaskConfig(
#     name="med_qa:cf",
#     prompt_function=med_qa_cf_prompt,
#     hf_repo="bigbio/med_qa",
#     ...
# )

TASKS_TABLE = [
    med_mcqa_mcf_em,
    med_mcqa_mcf,
    med_mcqa_cf,
    med_qa_titaneval_cf,
    med_qa_titaneval_mcf,
    med_qa_titaneval_mcf_em,
]
