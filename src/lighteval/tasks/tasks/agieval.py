"""
name:
Agieval

dataset:
ruixiangcui/AGIEval @ 84ab72d9 (`data/v1`, the 9 official English multiple-choice files; the base and chat tasks), lighteval/agi_eval_en (only the `:cot` inspect variant)

abstract:
AGIEval is a human-centric benchmark specifically designed to evaluate the
general abilities of foundation models in tasks pertinent to human cognition and
problem-solving. This benchmark is derived from 20 official, public, and
high-standard admission and qualification exams intended for general human
test-takers, such as general college admission tests (e.g., Chinese College
Entrance Exam (Gaokao) and American SAT), law school admission tests, math
competitions, lawyer qualification tests, and national civil service exams.

languages:
english

tags:
biology, chemistry, geography, history, knowledge, language, multiple-choice, physics, reasoning

paper:
https://arxiv.org/abs/2304.06364
"""

import hashlib
import json
import os
import re
from functools import partial
from pathlib import Path
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.agieval_demos import AGIEVAL_DEMOS


_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def record_to_sample(record):
    """Read the English AGIEval schema, preserving passage context and option text."""
    labels = ascii_uppercase[: len(record["options"])]
    if not labels or len(record["options"]) > len(ascii_uppercase) or record["label"] not in list(labels):
        raise ValueError("AGIEval multiple-choice records must have options and a valid answer letter.")

    options = record["options"]
    option_labels = []
    for option in options:
        match = re.match(r"^\s*(?:\(([A-Z])\)|([A-Z])(?:[.)]|\s+))", option)
        option_labels.append((match[1] or match[2]) if match else None)
    if all(option_labels):
        if set(option_labels) != set(labels):
            raise ValueError("AGIEval option labels must be unique and cover the available choices.")
        # Some LogiQA rows store B and C out of order. Golds refer to these
        # original option labels, so restore label order before relabeling.
        options = [option for _, option in sorted(zip(option_labels, options))]

    choices = [
        re.sub(rf"^\s*(?:\({label}\)\s*|{label}(?:[.)]\s*|\s+))", "", option, count=1).strip()
        for label, option in zip(labels, options)
    ]
    question = f"Question: {record['question'].strip()}"
    passage = (record.get("passage") or "").strip()
    if passage:
        question = f"Passage:\n{passage}\n\n{question}"
    return Sample(input=question, choices=choices, target=record["label"])


def sample_to_fewshot(sample):
    options = "\n".join(f"{label}. {option}" for label, option in zip(ascii_uppercase, sample.choices))
    return f"{sample.input}\n\n{options}\n\nAnswer: {sample.target}"


_SAT_CONFLICT_QUESTION = "Question: Throughout the passage, the narrator is portrayed as someone who is"
_SAT_CONFLICT_CHOICES = [
    "reserved around unfamiliar people.",
    "attuned to her immediate surroundings.",
    "sympathetic to the needs of others.",
    "anxious about her responsibilities.",
]


def _is_valid_sample(sample: Sample) -> bool:
    # SAT-English train rows 100 and 101 have identical content but golds B and
    # A. Exclude both, matching content so shuffling cannot change the filter.
    # The same two rows are in `sat-en-without-passage` (rows 100 and 101 of that file,
    # same question and options, golds B and A, no passage): excluded there too (owner 11-16).
    has_conflict_text = (
        sample.input.endswith("\n\n" + _SAT_CONFLICT_QUESTION) or sample.input == _SAT_CONFLICT_QUESTION
    ) and sample.choices == _SAT_CONFLICT_CHOICES
    with_passage = sample.input.startswith(
        "Passage:\nAt last, Old Widow Lau was done haggling with the driver and we stepped inside Father's shop."
    )
    without_passage = sample.input == _SAT_CONFLICT_QUESTION
    return not (has_conflict_text and (with_passage or without_passage))


def agieval_prompt_cot_eng(line, task_name: str = None):
    sample = record_to_sample(line)
    if not _is_valid_sample(sample):
        return None
    labels = ascii_uppercase[: len(sample.choices)]
    instruction = (
        "Answer the following multiple choice question. The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
        f"{labels}. Think step by step before answering."
    )
    options = "\n".join(f"{label}. {option}" for label, option in zip(labels, sample.choices))
    query = f"{instruction}\n\n{sample.input}\n\n{options}\n\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {label}" for label in labels],
        gold_index=labels.index(sample.target),
        instruction=instruction,
    )


def agieval_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled options in prompt, score label tokens via logprobs."""
    sample = record_to_sample(line)
    if not _is_valid_sample(sample):
        return None
    labels = ascii_uppercase[: len(sample.choices)]
    instruction = "The following are multiple choice questions (with answers).\n\n"
    options = "\n".join(f"{label}. {option}" for label, option in zip(labels, sample.choices))
    gold_index = labels.index(sample.target)
    return Doc(
        task_name=task_name,
        query=f"{instruction}{sample.input}\n\n{options}\n\nAnswer:",
        choices=[f" {label}" for label in labels],
        gold_index=gold_index,
        fewshot_sorting_class=sample.choices[gold_index],
        instruction=instruction,
    )


# The hub dataset `lighteval/agi_eval_en` has only these 7 English subsets (and `math`); it is read by the `:cot` inspect variant alone.
AGIEVAL_HUB_SUBSETS = [
    "aqua_rat",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
    "sat-math",
]

# lighteval/agi_eval_en: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "8670dcc6a5aa2c5e177b14905ec892df983446cd"

# ---- the OFFICIAL AGIEval data (owner 11-16: all nine English multiple-choice files, the base and the chat tasks) -----------------------------------------------------------------------------
# github.com/ruixiangcui/AGIEval, `data/v1/<file>.jsonl`, commit 84ab72d94318290aad2e4ec820d535a95a1f7552 (HEAD 2026-11-16; the nine files are byte-identical in `data/v1` and `data/v1_1`
# and equal to commit 341944036cb4cb02da5bc25fe9752dbeed5ac9f6, the commit OLMES copied: resources/olmes/oe_eval/dependencies/AGIEval/CHANGELOG.md). The nine are the English multiple-choice tasks of
# the official `english_qa_datasets` (src/dataset_loader.py / agi_dataset.py l.14-15) and of OLMES' `AGI_EVAL_ENGLISH_DATASETS` (oe_eval/data/agi_eval_tasks.py).
# Licence: MIT (Microsoft) for the SAT / Gaokao files, MIT for LSAT, Apache-2.0 for AQuA-RAT, CC BY-NC-SA 4.0 for LogiQA (data/v1/LICENSE).
# The files are read from <cache>/xdg/agieval_v1/ (titan-eval prepare agieval, login node; a `kind: files` row of configs/datasets.lock.yaml), the sha256 below is checked on every load.
AGIEVAL_REPO = "agieval_local"          # the `hf_repo` of the tasks; `load_datasets` reads the file `hf_subset` with `load_agieval`
AGIEVAL_COMMIT = "84ab72d94318290aad2e4ec820d535a95a1f7552"
AGIEVAL_FOLDER = "agieval_v1"
AGIEVAL_FILES = {           # official file -> (rows, sha256)
    "lsat-ar": (230, "3b3e3fe09a07c695326adb82f38da0d67d5bbaadab41551f198c3102f1ea9dc8"),
    "lsat-lr": (510, "c6acb4d843db7b515da4d853e52bb8e6e5f776910f2e35a1805634db24bf94ea"),
    "lsat-rc": (269, "0eed491b3099d66b8110d4fabb98ce43285a6ac61ea5643285eec11b8abce202"),
    "logiqa-en": (651, "63d0e8efaca1944e7eac5037903650f92ca9c036155a66b4e95bf2aa06da1702"),
    "sat-math": (220, "9cde4c0522b6196852a5562db6e72bf8817b4802822db0fc10013d1becdab3c3"),
    "sat-en": (206, "33fe87b32ff16ae7ba52b27bb398190b082253b3617a3eb9737005262ba12dd4"),
    "aqua-rat": (254, "cff42e946e6082dacb27285dae19cb4be98408ab760fe43c6c462e543be50572"),
    "sat-en-without-passage": (206, "77eb57bb6f6f39466d5d169de5253e77755ef521ad6692f582307672affbf593"),
    "gaokao-english": (306, "2de1b1e5d9718d908ffb46665e949bff83bef956ad8b7de16ea412e058162b01"),
}
# The task-name token of each file: the hub spelling `aqua_rat` is kept for aqua-rat (so the subtask names of earlier results stay), the others are the file names.
AGIEVAL_SUBSETS = ["aqua_rat", "gaokao-english", "logiqa-en", "lsat-ar", "lsat-lr", "lsat-rc", "sat-en", "sat-en-without-passage", "sat-math"]


def agieval_file(subset: str) -> str:
    """The official file name of a task-name token (`aqua_rat` -> `aqua-rat`)."""
    return subset.replace("_", "-")


def agieval_dir() -> Path:
    """<cache>/xdg/agieval_v1: $TITAN_EVAL_CACHE, else the parent of $HF_HOME (as the other prepared files, chat_math2.imo_dir)."""
    root = os.environ.get("TITAN_EVAL_CACHE") or (str(Path(os.environ["HF_HOME"]).parent) if os.environ.get("HF_HOME") else "")
    return Path(root or ".") / "xdg" / AGIEVAL_FOLDER


def load_agieval(subset: str):
    """The rows of one official file as a DatasetDict with ONE split, `train` (the evaluated one, as the hub copy). The file's sha256 is checked against the pin first; only the fields the
    tasks read are kept (passage, question, options, label; the `other` field has a different shape in each file)."""
    from datasets import Dataset, DatasetDict

    name = agieval_file(subset)
    rows, digest = AGIEVAL_FILES[name]
    path = agieval_dir() / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: AGIEval file {name} (ruixiangcui/AGIEval @ {AGIEVAL_COMMIT[:8]}) not prepared; on the login node: titan-eval prepare agieval")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise ValueError(f"{path}: sha256 {got} is not the pinned {digest} (ruixiangcui/AGIEval @ {AGIEVAL_COMMIT})")
    data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(data) != rows:
        raise ValueError(f"{path}: {len(data)} rows, the pinned file has {rows}")
    keep = [{"passage": r.get("passage") or "", "question": r["question"], "options": list(r["options"]), "label": r["label"]} for r in data]
    return DatasetDict({"train": Dataset.from_list(keep)})


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"agieval_eng:{subset}:cot",
        sample_fields=record_to_sample,
        filter=_is_valid_sample,
        sample_to_fewshot=sample_to_fewshot,
        solver=[multiple_choice(cot=True, max_tokens=32768)],
        scorer=choice(),
        prompt_function=agieval_prompt_cot_eng,
        hf_repo="lighteval/agi_eval_en",
        hf_revision=_REVISION,
        hf_subset=subset,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["train"],
        # The Hub's LogiQA validation split contains AQUA questions. Sample genuine
        # LogiQA examples instead; the few-shot sampler excludes the evaluated row.
        few_shots_split="train" if subset == "logiqa-en" else "validation",
        few_shots_select="random_sampling",
        generation_size=32768,
        metrics=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Question:"],
        version=4,
    )
    for subset in AGIEVAL_HUB_SUBSETS
]

def _agieval_demo_rows(subset: str) -> list[dict]:
    """The official AGIEval demos of a subset (3 or 5), in the official order (agieval_demos.py)."""
    return AGIEVAL_DEMOS[subset.replace("_", "-")]


# The mcf / mcf_em variants take their shots ONLY from the official demos: the first K of them, in order, K =
# min(requested K, official count) (3 for lsat-ar, lsat-lr, lsat-rc, logiqa-en, sat-en, sat-en-without-passage, gaokao-english; 5 for sat-math, aqua_rat).
# They are never sampled from the evaluated split. A smaller pool than K logs a warning (prompt_manager.py) and the
# `n_shots` metric / the INFO line of `get_docs` give the shots really used. A demo identical to the evaluated item
# would be excluded (none is: checked passage + question + options).
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"agieval_eng:{subset}:mcf",
        prompt_function=agieval_mcf_prompt,
        hf_repo=AGIEVAL_REPO,
        hf_revision=AGIEVAL_COMMIT,
        hf_subset=agieval_file(subset),
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_rows=partial(_agieval_demo_rows, subset),
        few_shots_select="sequential",
        generation_size=-1,
        metrics=_MCF_METRICS,
        stop_sequence=["\n"],
        version=2,          # 1 -> 2 (owner 11-16): the official AGIEval v1 files, 9 English subsets (gaokao-english and sat-en-without-passage added)
    )
    for subset in AGIEVAL_SUBSETS
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"agieval_eng:{subset}:mcf_em",
        prompt_function=agieval_mcf_prompt,
        hf_repo=AGIEVAL_REPO,
        hf_revision=AGIEVAL_COMMIT,
        hf_subset=agieval_file(subset),
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_rows=partial(_agieval_demo_rows, subset),
        few_shots_select="sequential",
        generation_size=5,
        metrics=[Metrics.exact_match, Metrics.n_shots],
        stop_sequence=["\n"],
        version=3,          # 2 -> 3 (owner 11-16): the official AGIEval v1 files, 9 English subsets (gaokao-english and sat-en-without-passage added)
    )
    for subset in AGIEVAL_SUBSETS
]
