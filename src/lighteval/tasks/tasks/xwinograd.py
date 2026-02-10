"""
name:
Xwinograd

dataset:
Muennighoff/xwinograd

abstract:
Multilingual winograd schema challenge as used in Crosslingual Generalization through Multitask Finetuning.

languages:
english, french, japanese, portuguese, russian, chinese

tags:
commonsense, multilingual, reasoning

paper:
https://arxiv.org/abs/2211.01786
"""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import MCFFormulation
from lighteval.utils.language import Language


xwinograd_instruction = "Fill in the blank with the correct option."

XWINOGRAD_SUBSETS = {
    "en": Language.ENGLISH,
    "fr": Language.FRENCH,
    "jp": Language.JAPANESE,
    "pt": Language.PORTUGUESE,
    "ru": Language.RUSSIAN,
    "zh": Language.CHINESE,
}

def xwinograd_prompt(line, task_name: str = None):
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,
    )

def xwinograd_mcf_adapter(line):
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    gold_idx = int(line["answer"]) - 1 if line["answer"] != "" else -1
    return {
        "question": query,
        "choices": [f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        "gold_idx": gold_idx,
    }

xwinograd_tasks = [
    LightevalTaskConfig(
        name=f"xwinograd:{subset}",
        prompt_function=xwinograd_prompt,
        hf_repo="Muennighoff/xwinograd",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        metrics=[
            LogLikelihoodAccMetric(),
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        ],
        stop_sequence=["\n"],
        version=0,
    )
    for subset in XWINOGRAD_SUBSETS.keys()
]

xwinograd_mcf_tasks = [
    LightevalTaskConfig(
        name=f"xwinograd_mcf:{subset}",
        prompt_function=get_mcq_prompt_function(lang, xwinograd_mcf_adapter, MCFFormulation()),
        hf_repo="Muennighoff/xwinograd",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        # metrics=[Metrics.exact_match],
        metrics=[LogLikelihoodAccMetric()],
        # generation_size=16,
        # stop_sequence=["\n"],
        version=0,
    )
    for subset, lang in XWINOGRAD_SUBSETS.items()
]

TASKS_TABLE = [
    *xwinograd_tasks,
    *xwinograd_mcf_tasks,
]
