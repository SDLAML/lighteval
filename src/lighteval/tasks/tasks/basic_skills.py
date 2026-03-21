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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Assumed dataset fields: "question" (str), "answer" (str)
# Adjust if the actual allenai/basic-skills schema differs.

# HF config names in allenai/basic-skills (only validation split available)
_BASIC_SKILLS_SUBSETS = [
    "arithmetic",
    "string_operations",
    "coding",
    "logical_reasoning",
    "common_knowledge",
    "pattern",
]


def basic_skills_prompt(line, task_name: str = None):
    """Greedy variant: generate answer text, compare with exact match."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + line["answer"]],
        gold_index=0,
    )


def basic_skills_bpb_prompt(line, task_name: str = None):
    """BPB variant: CF-style prompt with gold answer as single choice."""
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


TASKS_TABLE = []

for _subset in _BASIC_SKILLS_SUBSETS:
    # CF variant: RC per-token normalization + BPB on gold (merged)
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"basic_skills:{_subset}:cf",
            prompt_function=basic_skills_bpb_prompt,
            hf_repo="allenai/basic-skills",
            hf_subset=_subset,
            hf_avail_splits=["validation"],
            evaluation_splits=["validation"],
            few_shots_split="validation",
            few_shots_select="random_sampling",
            generation_size=-1,
            metrics=[
                Metrics.target_bits_per_byte,
            ],
            stop_sequence=["\n"],
            version=0,
        )
    )
