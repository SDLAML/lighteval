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
    """Prompt = task description + function header; gold = full code with leading space.

    Matches OLMO's default MBPP format: query ends at the function header colon
    so the model predicts the entire function (signature + body) as the gold.
    Leading space matches OLMO's perplexity_leading_space=True.
    """
    query = line["prompt"] + line["code"].split(":")[0] + ":"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + line["code"]],
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
