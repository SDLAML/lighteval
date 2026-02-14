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

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

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


def mmlu_prompt(line, task_name: str = None):
    subject = line["subject"]
    query = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\nQuestion: {line['question']}"
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])])
    query += "\nAnswer:"

    gold_ix = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

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
    query = line['question']
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])])
    query += "\nThink step by step before answering."

    gold_ix = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n",
    )

def mmlu_redux_prompt(line, task_name: str = None):
    if line['error_type'] != "ok":
        return None
    
    query = f"Question: {line['question']}"
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])])
    query += "\n\nAnswer:"

    gold_ix = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
    )

def mmlu_redux_chat_prompt(line, task_name: str = None):
    if line['error_type'] != "ok":
        return None
    
    query = line['question']
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])])
    query += "\n\nThink step by step before answering."

    gold_ix = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        # fewshot_sorting_class=line["choices"][gold_ix],
    )

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu:{subset}",
        prompt_function=mmlu_prompt,
        hf_repo="lighteval/mmlu",
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

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_chat:{subset}",
        prompt_function=mmlu_chat_prompt,
        hf_repo="lighteval/mmlu",
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=4096,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Question:"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_redux:{subset}",
        prompt_function=mmlu_redux_prompt,
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=5,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_redux_chat:{subset}",
        prompt_function=mmlu_redux_chat_prompt,
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=4096,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Question:"],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]
