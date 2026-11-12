"""
name:
Gpqa

dataset:
Idavidrein/gpqa

abstract:
GPQA is a dataset of 448 expert-written multiple-choice questions in biology,
physics, and chemistry, designed to test graduate-level reasoning. The questions
are extremely difficult—PhD-level experts score about 65%, skilled non-experts
34% (even with web access), and GPT-4 around 39%. GPQA aims to support research
on scalable oversight, helping humans evaluate and trust AI systems that may
exceed human expertise.

languages:
english

tags:
biology, chemistry, graduate-level, multiple-choice, physics, qa, reasoning, science

paper:
https://arxiv.org/abs/2311.12022
"""

import ast
import hashlib
import os
import random
from functools import lru_cache
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

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def _stable_choices(line):
    """Deterministic shuffle of [Correct, Incorrect1, Incorrect2, Incorrect3] by question hash."""
    items = [
        line["Correct Answer"],
        line["Incorrect Answer 1"],
        line["Incorrect Answer 2"],
        line["Incorrect Answer 3"],
    ]
    seed = int(hashlib.md5(line["Question"].encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(items)
    gold = items.index(line["Correct Answer"])
    return items, gold


def record_to_sample(record):
    gold_index = random.randint(0, 3)
    choices = [record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
    choices.insert(gold_index, record["Correct Answer"])
    return Sample(
        input=record["Question"].strip(),
        choices=choices,
        target=ascii_uppercase[gold_index],
    )


def sample_to_fewshot(sample):
    return f"{sample.input}\n\n" + f"ANSWER: {sample.target}"


def gpqa_prompt(line, task_name: str = None):
    GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[: len(choices)],
        gold_index=gold_index,
        instruction=query,
    )


def gpqa_instruct_prompt(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    instruction = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
    query_template = "{Instruction}\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        Question=line["Question"].strip(),
        Instruction=instruction,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


gpqa = LightevalTaskConfig(
    name="gpqa:mc",
    prompt_function=gpqa_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

gpqa_diamond_instruct = LightevalTaskConfig(
    name="gpqa:diamond",
    prompt_function=gpqa_instruct_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 1})],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=1,
)

gpqa_extended_instruct = LightevalTaskConfig(
    name="gpqa:extended",
    prompt_function=gpqa_instruct_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_extended",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=0,
)

gpqa_main_instruct = LightevalTaskConfig(
    name="gpqa:main",
    prompt_function=gpqa_instruct_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=0,
)

def gpqa_cf_prompt(line, task_name: str = None):
    """CF variant: score full answer texts via logprobs (deterministic shuffle)."""
    choices, gold = _stable_choices(line)
    return Doc(
        task_name=task_name,
        query=f"Question: {line['Question'].strip()}\nAnswer:",
        choices=[" " + c for c in choices],
        gold_index=gold,
    )


def gpqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C/D options, score label tokens via logprobs."""
    choices, gold = _stable_choices(line)
    options = "\n".join(f" {l}. {c}" for l, c in zip(ascii_uppercase, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['Question'].strip()}\n{options}\nAnswer:",
        choices=[f" {l}" for l in ascii_uppercase[:4]],
        gold_index=gold,
    )


# The dataset revision these tasks are pinned to (gated dataset, cached by `titan-eval prefetch`).
_GPQA_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"


@lru_cache(maxsize=1)
def _gpqa_official_demos() -> list[dict]:
    """The 5 official GPQA few-shot examples, in the official order (OLMES FEWSHOT_SOURCES["Original:GPQA"]).

    They have the fields of a GPQA row (Question, Correct Answer, Incorrect Answer 1-3; the explanation is not used) and are
    shown by `gpqa_mcf_prompt` like an evaluated row. GPQA asks that its questions are not posted online in plain text
    (canary string), so they are NOT copied into this repository: they are read at run time from the OLMES clone
    (`resources/olmes/oe_eval/tasks/fewshot_sources.py` next to this clone, or the file named by $GPQA_FEWSHOT_FILE) by
    parsing that file (`ast.literal_eval` of the "Original:GPQA" entry, nothing is executed).
    """
    # parents[5] = the `resources/` folder that holds this clone (resources/lighteval/src/lighteval/tasks/tasks/gpqa.py)
    default = Path(__file__).resolve().parents[5] / "olmes" / "oe_eval" / "tasks" / "fewshot_sources.py"
    path = Path(os.environ.get("GPQA_FEWSHOT_FILE") or default)
    if not path.is_file():
        raise FileNotFoundError(
            f"GPQA few-shot examples: {path} not found. Clone allenai/olmes into resources/olmes or set GPQA_FEWSHOT_FILE "
            "to its oe_eval/tasks/fewshot_sources.py."
        )
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "Original:GPQA"
        ):
            demos = ast.literal_eval(node.value)
            assert len(demos) == 5, f"expected the 5 official GPQA demos, found {len(demos)}"
            return [dict(demo) for demo in demos]
    raise ValueError(f'FEWSHOT_SOURCES["Original:GPQA"] not found in {path}')


# Shots = the 5 official GPQA examples, the first K in order for every prompt (|5 = all five), never rows of the evaluated
# split. A demo equal to the evaluated question is excluded (one of the 5 is a question of the main and the diamond set:
# that single item gets 4 shots).
_GPQA_COMMON = dict(
    hf_repo="Idavidrein/gpqa",
    hf_revision=_GPQA_REVISION,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_rows=_gpqa_official_demos,
    few_shots_select="sequential",
    stop_sequence=["\n"],
)

gpqa_diamond_cf = LightevalTaskConfig(
    name="gpqa:diamond:cf",
    prompt_function=gpqa_cf_prompt,
    hf_subset="gpqa_diamond",
    generation_size=-1,
    metrics=_CF_METRICS,
    version=0,
    **_GPQA_COMMON,
)

gpqa_diamond_mcf = LightevalTaskConfig(
    name="gpqa:diamond:mcf",
    prompt_function=gpqa_mcf_prompt,
    hf_subset="gpqa_diamond",
    generation_size=-1,
    metrics=_MCF_METRICS,
    version=0,
    **_GPQA_COMMON,
)

gpqa_diamond_mcf_em = LightevalTaskConfig(
    name="gpqa:diamond:mcf_em",
    prompt_function=gpqa_mcf_prompt,
    hf_subset="gpqa_diamond",
    generation_size=1,
    metrics=[Metrics.exact_match],
    version=0,
    **_GPQA_COMMON,
)

gpqa_main_mcf_em = LightevalTaskConfig(
    name="gpqa:main:mcf_em",
    prompt_function=gpqa_mcf_prompt,
    hf_subset="gpqa_main",
    generation_size=1,
    metrics=[Metrics.exact_match],
    version=0,
    **_GPQA_COMMON,
)

TASKS_TABLE = [
    gpqa,
    gpqa_diamond_instruct,
    gpqa_extended_instruct,
    gpqa_main_instruct,
    gpqa_diamond_cf,
    gpqa_diamond_mcf,
    gpqa_diamond_mcf_em,
    gpqa_main_mcf_em,
]
