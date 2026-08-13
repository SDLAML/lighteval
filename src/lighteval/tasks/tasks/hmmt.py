"""
name:
Hmmt

dataset:
MathArena/hmmt_feb_2026

abstract:
The Harvard-MIT Mathematics Tournament (HMMT) is a highly challenging
mathematics competition for high-school students, organized by students at
Harvard and MIT. The tournament features problems across areas such as algebra,
number theory, combinatorics, and geometry, and includes both individual and
team-based rounds. The problems require exact mathematical answers and test
advanced competition mathematics, problem solving, and multi-step reasoning.

languages:
english

tags:
math, reasoning

paper:
https://www.hmmt.org/

starred:
true
"""

from textwrap import dedent

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_PROMPT_TEMPLATE = dedent("""
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{prompt}
""")


def normalize_answer(answer: str) -> str:
    """Wrap HMMT's bare LaTeX gold answers for extractive math scoring."""
    return f"${answer}$"


def record_to_sample(record):
    return Sample(input=record["problem"], target=normalize_answer(record["answer"]))


def hmmt_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["problem"]),
        choices=[normalize_answer(line["answer"])],
        gold_index=0,
    )

hmmt_feb_2026 = LightevalTaskConfig(
    name="hmmt_feb_2026",
    prompt_function=hmmt_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="MathArena/hmmt_feb_2026",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 10})],
    version=1,
)

TASKS_TABLE = [
    hmmt_feb_2026,
]
