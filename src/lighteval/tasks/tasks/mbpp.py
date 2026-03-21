"""
MBPP BPB evaluation task for lighteval.

Dataset : google-research-datasets/mbpp, subset "sanitized"
ICL     : 3 shots sampled from the train split
Metric  : BPB = -log2 p(code | prompt) / bytes(code)

The prompt is a docstring-style block containing the task description and
the assert-based test cases.  The gold continuation is the full reference
solution (starting with the `def` line).

Task name: mbpp:bpb
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mbpp_bpb_prompt(line, task_name=None):
    """Build a docstring prompt; gold = reference solution code."""
    tests = "\n".join(line.get("test_list", []))
    # Format as a Python docstring block so the model sees:
    #   """
    #   <task description>
    #   <assert statements>
    #   """
    #   def function_name(...):   ← start of gold continuation
    prompt = f'"""\n{line["prompt"]}\n{tests}\n"""\n'
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[line["code"]],
        gold_index=0,
    )


mbpp_bpb = LightevalTaskConfig(
    name="mbpp:bpb",
    prompt_function=mbpp_bpb_prompt,
    hf_repo="google-research-datasets/mbpp",
    hf_subset="sanitized",
    hf_avail_splits=["train", "validation", "test", "prompt"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=[Metrics.target_bits_per_byte],
    stop_sequence=None,
    version=0,
)

TASKS_TABLE = [mbpp_bpb]
