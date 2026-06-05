"""
name:
Basic Skills

dataset:
allenai/basic-skills

abstract:
Basic Skills is a benchmark covering fundamental cognitive and reasoning abilities,
including arithmetic, string manipulation, coding, logical reasoning, common sense,
and pattern recognition. Each subset tests a distinct basic skill category.

languages:
english

tags:
arithmetic, reasoning, coding, commonsense, basic-skills, qa

paper:
"""

import hashlib
import random
import threading
from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Dataset fields: "question" (str), "answer" (str) — no built-in choices.
# Distractors for :cf and :mcf are built incrementally from the answers that
# flow through the prompt function (including few-shot examples, which are
# processed first). No separate dataset load is needed.
_BASIC_SKILLS_ACCUM: dict = {}   # {subset: set of answers}
_ACCUM_LOCK = threading.Lock()

# HF config names in allenai/basic-skills (only validation split available)
_BASIC_SKILLS_SUBSETS = [
    "arithmetic",
    "string_operations",
    "coding",
    "logical_reasoning",
    "common_knowledge",
    "pattern",
]


def _get_distractor_pool(subset: str, gold: str) -> list:
    """Return available distractors (excluding *gold*), then accumulate *gold*.

    The pool is thread-safe and grows as examples pass through the prompt function.
    Early examples in a subset may have fewer than 3 distractors until the pool
    accumulates enough unique answers from prior few-shot or test examples.
    """
    with _ACCUM_LOCK:
        current = list(_BASIC_SKILLS_ACCUM.get(subset, set()))
        _BASIC_SKILLS_ACCUM.setdefault(subset, set()).add(gold)
    return [a for a in current if a != gold]


def basic_skills_bpb_prompt(line, task_name: str = None):
    """BPB variant: single-choice, score gold continuation only (no acc ranking)."""
    gold_text = line["answer"]
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


def basic_skills_cf_prompt(line, task_name: str = None):
    """CF variant: gold answer vs up to 3 distractors (RC_per-token acc + BPB).

    Subset is inferred from task_name (e.g. 'basic_skills:arithmetic:cf' → 'arithmetic').
    Distractors are drawn from a module-level pool that accumulates answers from
    examples as they are processed — no separate dataset load required.
    """
    if not line.get("answer") or not line["answer"].strip():
        return None
    subset = task_name.split(":")[1] if task_name else _BASIC_SKILLS_SUBSETS[0]
    gold = line["answer"]
    seed = int(hashlib.md5((line["question"] + gold).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    candidates = _get_distractor_pool(subset, gold)
    distractors = rng.sample(candidates, min(3, len(candidates)))
    all_choices = distractors + [gold]
    rng.shuffle(all_choices)
    gold_ix = all_choices.index(gold)
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in all_choices],
        gold_index=gold_ix,
    )


def basic_skills_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C/D options with gold answer (MC Acc).

    Uses a different hash seed from :cf to produce an independent shuffle.
    """
    if not line.get("answer") or not line["answer"].strip():
        return None
    subset = task_name.split(":")[1] if task_name else _BASIC_SKILLS_SUBSETS[0]
    gold = line["answer"]
    seed = int(hashlib.md5((line["question"] + gold).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed + 1)
    candidates = _get_distractor_pool(subset, gold)
    distractors = rng.sample(candidates, min(3, len(candidates)))
    all_choices = distractors + [gold]
    rng.shuffle(all_choices)
    gold_ix = all_choices.index(gold)
    query = f"Question: {line['question']}\n"
    query += "".join(f" {key}. {choice}\n" for key, choice in zip(ascii_uppercase, all_choices))
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + key for key in ascii_uppercase[:len(all_choices)]],
        gold_index=gold_ix,
    )


TASKS_TABLE = []

for _subset in _BASIC_SKILLS_SUBSETS:
    # BPB variant: single-choice gold continuation, BPB only
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"basic_skills:{_subset}:bpb",
            prompt_function=basic_skills_bpb_prompt,
            hf_repo="allenai/basic-skills",
            hf_subset=_subset,
            hf_avail_splits=["validation"],
            evaluation_splits=["validation"],
            few_shots_split="validation",
            few_shots_select="random_sampling",
            generation_size=-1,
            metrics=[Metrics.target_bits_per_byte],
            stop_sequence=["\n"],
            version=0,
        )
    )

    # CF variant: multi-choice RC_per-token acc + BPB (gold vs up to 3 distractors)
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"basic_skills:{_subset}:cf",
            prompt_function=basic_skills_cf_prompt,
            hf_repo="allenai/basic-skills",
            hf_subset=_subset,
            hf_avail_splits=["validation"],
            evaluation_splits=["validation"],
            few_shots_split="validation",
            few_shots_select="random_sampling",
            generation_size=-1,
            metrics=[
                LogLikelihoodAccMetric(),
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            ],
            stop_sequence=["\n"],
            version=0,
        )
    )

    # MCF variant: labeled A/B/C/D, MC acc (gold vs up to 3 distractors)
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"basic_skills:{_subset}:mcf",
            prompt_function=basic_skills_mcf_prompt,
            hf_repo="allenai/basic-skills",
            hf_subset=_subset,
            hf_avail_splits=["validation"],
            evaluation_splits=["validation"],
            few_shots_split="validation",
            few_shots_select="random_sampling",
            generation_size=-1,
            metrics=[
                LogLikelihoodAccMetric(),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
            stop_sequence=["\n"],
            version=0,
        )
    )
