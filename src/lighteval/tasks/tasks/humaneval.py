"""
HumanEval BPB evaluation task for lighteval.

Dataset : openai/openai_humaneval (HuggingFace)
ICL     : 3 shots sampled from the test split (no official train split)
Metric  : BPB = -log2 p(canonical_solution | prompt) / bytes(canonical_solution)

Task name: humaneval:bpb
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def humaneval_bpb_prompt(line, task_name=None):
    """Prompt = function signature + docstring; gold = canonical solution body."""
    return Doc(
        task_name=task_name,
        # `prompt` already ends at the opening of the function body (after the
        # closing triple-quote of the docstring), ready for the solution to follow.
        query=line["prompt"],
        choices=[line["canonical_solution"]],
        gold_index=0,
    )


humaneval_bpb = LightevalTaskConfig(
    name="humaneval:bpb",
    prompt_function=humaneval_bpb_prompt,
    hf_repo="openai/openai_humaneval",
    hf_subset="openai_humaneval",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    # No dedicated train split; lighteval will sample 3 ICL examples from the
    # test set (excluding the current example).
    few_shots_split="test",
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=[Metrics.target_bits_per_byte],
    stop_sequence=None,
    version=0,
)

TASKS_TABLE = [humaneval_bpb]
