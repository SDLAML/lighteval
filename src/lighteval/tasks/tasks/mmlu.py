"""
name:
Mmlu

dataset:
lighteval/mmlu

abstract:
MMMLU is a benchmark of general-knowledge and English language understanding.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2009.03300
"""

from functools import partial
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig, load_dataset_pinned
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

_MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def mmlu_cf_prompt(line, task_name: str = None):
    """CF variant: completion-format prompt with full answer texts as choices."""
    subject = line["subject"]
    query = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\nQuestion: {line['question']}\nAnswer:"

    gold_ix = (
        ascii_uppercase.index(line["answer"])
        if isinstance(line["answer"], str)
        else line["answer"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + c for c in line["choices"]],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n",
    )


def mmlu_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options in prompt, score label tokens via logprobs."""
    subject = line["subject"]
    query = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\nQuestion: {line['question']}"
    query += "".join(
        [f"\n {key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])]
    )
    query += "\nAnswer:"

    gold_ix = (
        ascii_uppercase.index(line["answer"])
        if isinstance(line["answer"], str)
        else line["answer"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n",
    )


def mmlu_chat_prompt(line, task_name: str = None):
    subject = line["subject"]
    query = line["question"]
    query += "".join(
        [f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])]
    )
    query += "\nThink step by step before answering."

    gold_ix = (
        ascii_uppercase.index(line["answer"])
        if isinstance(line["answer"], str)
        else line["answer"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n",
    )


def mmlu_redux_prompt(line, task_name: str = None, *, subject: str):
    """MCF redux prompt: the text of `mmlu_mcf_prompt` for the questions MMLU-Redux 2.0 marks "ok" (others -> None).

    The MMLU-Redux rows have no `subject` column (it is the config name), so the subject is passed in
    (`partial(mmlu_redux_prompt, subject=subset)`). The text comes from `mmlu_mcf_prompt` itself, so an item of this task
    and an item of `mmlu:<subject>:mcf_em` differ only in the question (the old copy had no space before the option letters).
    """
    if line["error_type"] != "ok":
        return None
    return mmlu_mcf_prompt(
        {
            "subject": subject,
            "question": line["question"],
            "choices": line["choices"],
            "answer": line["answer"],
        },
        task_name=task_name,
    )


# lighteval/mmlu: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_MMLU_REVISION = "31d46ab06e6934bb0d95f6918668716d1db6f921"
# edinburgh-dawg/mmlu-redux-2.0: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_MMLU_REDUX_2_0_REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"


def _mmlu_dev_rows(subset: str) -> list[dict]:
    """The dev questions of the ORIGINAL MMLU (`lighteval/mmlu`, 5 per subject), as MMLU-Redux-style rows.

    MMLU-Redux has no dev split; the redux tasks take their shots from here (not from the evaluated redux rows), so they
    get the same 5 demonstrations as `mmlu:<subject>:mcf_em`. Read from the cached / downloaded `lighteval/mmlu`.
    """
    dev = load_dataset_pinned("lighteval/mmlu", subset, revision=_MMLU_REVISION, split="dev")
    return [
        {"question": row["question"], "choices": row["choices"], "answer": row["answer"], "error_type": "ok"}
        for row in dev
    ]


def mmlu_redux_cf_prompt(line, task_name: str = None):
    """CF variant for redux — mirrors mmlu_cf_prompt exactly, filters non-ok rows."""
    if line["error_type"] != "ok":
        return None

    subject = line["subject"]
    gold_ix = (
        ascii_uppercase.index(line["answer"])
        if isinstance(line["answer"], str)
        else line["answer"]
    )

    query = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\nQuestion: {line['question']}\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + c for c in line["choices"]],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n",
    )


def mmlu_redux_chat_prompt(line, task_name: str = None):
    """Chat variant for redux — mirrors mmlu_chat_prompt exactly, filters non-ok rows."""
    if line["error_type"] != "ok":
        return None

    subject = line["subject"]
    query = line["question"]
    query += "".join(
        [f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])]
    )
    query += "\nThink step by step before answering."

    gold_ix = (
        ascii_uppercase.index(line["answer"])
        if isinstance(line["answer"], str)
        else line["answer"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n",
    )


# Greedy variant: MCF-style prompt, generate token(s), exact match
TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu:{subset}:mcf_em",
        prompt_function=mmlu_mcf_prompt,
        hf_repo="lighteval/mmlu",
        hf_revision=_MMLU_REVISION,
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=5,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# Chat/instruction-following variant (kept for compatibility)
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu:{subset}:chat",
        prompt_function=mmlu_chat_prompt,
        hf_repo="lighteval/mmlu",
        hf_revision=_MMLU_REVISION,
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=32768,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Question:"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# CF variant: completion-format with full answer texts, logprobs-based
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu:{subset}:cf",
        prompt_function=mmlu_cf_prompt,
        hf_repo="lighteval/mmlu",
        hf_revision=_MMLU_REVISION,
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=-1,
        metrics=_CF_METRICS,
        stop_sequence=["\n"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu:{subset}:mcf",
        prompt_function=mmlu_mcf_prompt,
        hf_repo="lighteval/mmlu",
        hf_revision=_MMLU_REVISION,
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# MMLU Redux — Greedy variant
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_redux:{subset}:mcf_em",
        prompt_function=partial(mmlu_redux_prompt, subject=subset),
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_revision=_MMLU_REDUX_2_0_REVISION,
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        # Shots = the 5 dev questions of the original MMLU, never rows of the evaluated redux split. `few_shots_select=None`
        # is "balanced": all 5 in the same seeded order as `mmlu:<subject>:mcf_em` (same prompt text, same shots).
        few_shots_rows=partial(_mmlu_dev_rows, subset),
        few_shots_select=None,
        generation_size=5,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=1,
    )
    for subset in _MMLU_SUBSETS
]

# MMLU Redux — Chat/instruction-following variant (kept for compatibility)
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_redux:{subset}:chat",
        prompt_function=mmlu_redux_chat_prompt,
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_revision=_MMLU_REDUX_2_0_REVISION,
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling_from_train",
        generation_size=4096,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Question:"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# MMLU Redux — CF variant
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_redux:{subset}:cf",
        prompt_function=mmlu_redux_cf_prompt,
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_revision=_MMLU_REDUX_2_0_REVISION,
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=_CF_METRICS,
        stop_sequence=["\n"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# MMLU Redux — MCF variant (TRUE MCF: logprob on label tokens)
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_redux:{subset}:mcf",
        prompt_function=partial(mmlu_redux_prompt, subject=subset),
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_revision=_MMLU_REDUX_2_0_REVISION,
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_rows=partial(_mmlu_dev_rows, subset),
        few_shots_select=None,
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=1,
    )
    for subset in _MMLU_SUBSETS
]
