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


_ANSWER_PREFIX = "Here is the completed function:\n\n```python\n"


def humaneval_bpb_prompt(line, task_name=None):
    """Prompt = function signature + answer_prefix; gold = canonical solution.

    Matches OLMO: query = prompt + answer_prefix so the gold continuation is
    scored in the context of the opening code fence, not the raw docstring end.
    Leading space is added to the gold to align with OLMO's perplexity_leading_space=True.
    """
    return Doc(
        task_name=task_name,
        query=line["prompt"] + _ANSWER_PREFIX,
        choices=[" " + line["canonical_solution"]],
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
